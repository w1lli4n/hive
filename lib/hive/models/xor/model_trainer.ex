defmodule Hive.Models.Xor.ModelTrainer do
  @behaviour Hive.Core.ModelTrainer
  use GenServer

  require Logger

  # --- API ---

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, :ok, name: __MODULE__)
  end

  @doc """
  Runs the model training process across available nodes.
  This function will block until the training is complete or an error occurs.
  """
  @impl Hive.Core.ModelTrainer
  def run(model, data, opts, id \\ nil, initial_model_state \\ Axon.ModelState.empty()) do
    # Send a cast to initiate training, and then wait for the result
    # We need a way to block the caller until training is truly done.
    # A GenServer.call is synchronous, but the *internal* work is async.
    # So, we'll send a call to start, and the GenServer will manage the async loop.
    # When it's done, it will reply to a specific monitor or named process.

    # A better approach for a blocking API:
    # 1. Send a call to start the training.
    # 2. The ModelTrainer GenServer will then manage the epochs internally.
    # 3. When training is complete, the ModelTrainer can then reply to the original caller
    #    with the final model state. This implies a longer-running GenServer.call.

    # Let's simplify and make the GenServer manage the entire loop:

    # The caller will block on this GenServer.call until the entire training is done.
    GenServer.call(
      __MODULE__,
      {:start_full_training, model, data, opts, id, initial_model_state},
      :infinity
    )
  end

  @doc """
  Merges a list of Axon model states by averaging their weights.
  """
  @impl Hive.Core.ModelTrainer
  def merge_models(model_states, initial_model_state) do
    Enum.reduce(model_states, initial_model_state, fn ms, acc ->
      if acc == Axon.ModelState.empty() do
        ms
      else
        Axon.ModelState.merge(ms, acc, &average/3)
      end
    end)
  end

  # --- GenServer Callbacks ---

  @impl true
  def init(_args) do
    :net_kernel.monitor_nodes(true)

    {:ok,
     %{
       nodes: Node.list() ++ [Node.self()],
       # Store active training runs by their ID
       training_runs: %{},
       next_model_id: 0
     }}
  end

  # This handle_call initiates the entire training process
  @impl true
  def handle_call(
        {:start_full_training, model, data, opts, model_id, initial_model_state},
        from,
        state
      ) do
    # Sync call for initial ID
    generated_id = model_id || get_next_model_idx_sync(state)

    # Store the caller's PID so we can reply when training is truly complete
    # Update next_model_id immediately
    updated_state = %{state | next_model_id: generated_id + 1}

    # Initialize the training run entry in the state
    training_run_state = %{
      # The process to reply to when done
      caller: from,
      model: model,
      data: data,
      opts: opts,
      current_model_state: initial_model_state,
      remaining_epochs: opts[:epochs],
      active_tasks: [],
      completed_model_states: [],
      # Track progress for logging
      current_step: 0
    }

    updated_training_runs = Map.put(updated_state.training_runs, generated_id, training_run_state)
    new_state = %{updated_state | training_runs: updated_training_runs}

    # Immediately trigger the first training step
    {:noreply, new_state, {:continue, {:start_step, generated_id}}}
  end

  # This handle_call is for internal ID generation, not part of the main training loop
  @impl true
  def handle_call(:get_next_model_id_sync, _from, state) do
    %{next_model_id: next_model_id} = state
    {:reply, {:ok, next_model_id}, %{state | next_model_id: next_model_id + 1}}
  end

  # handle_continue is used to perform actions immediately after handle_call/init
  @impl true
  def handle_continue({:start_step, model_id}, state) do
    %{training_runs: training_runs, nodes: nodes} = state

    %{
      model: model,
      data: data,
      opts: opts,
      current_model_state: current_model_state,
      remaining_epochs: remaining_epochs,
      # Should be empty at start of step
      active_tasks: _active_tasks,
      # Should be empty at start of step
      completed_model_states: _completed_model_states,
      caller: _caller
    } = Map.fetch!(training_runs, model_id)

    if Enum.empty?(nodes) do
      Logger.warning("No nodes available for training for model ID #{model_id}.")
      # Reply with error to the caller, or handle according to desired behavior
      GenServer.reply(Map.fetch!(training_runs, model_id).caller, {:error, "no_nodes_available"})
      # Clean up
      {:noreply, %{state | training_runs: Map.delete(training_runs, model_id)}}
    else
      # Check if enough epochs remain for at least one step
      if remaining_epochs >= opts[:steps] do
        Logger.info(
          "Starting training step for model ID: #{model_id}, remaining epochs: #{remaining_epochs}"
        )

        # Distribute tasks to nodes
        current_step_tasks =
          Enum.map(nodes, fn node ->
            # Monitor the task to get its result
            Task.Supervisor.async_nolink(
              # Use correct supervisor for the node
              {Hive.TaskSupervisor, node},
              Hive.Models.Xor.FragmentTrainer,
              :run,
              [model, data, opts, model_id, current_model_state]
            )
          end)

        # Update the training run state with active tasks
        updated_training_run =
          Map.update!(training_runs, model_id, fn run ->
            %{
              run
              | active_tasks: current_step_tasks,
                completed_model_states: [],
                current_step: run.current_step + 1
            }
          end)

        {:noreply,
         %{state | training_runs: Map.put(training_runs, model_id, updated_training_run)}}
      else
        # Training is complete (or not enough epochs for another full step)
        # This branch should ideally be reached after all steps have completed.
        Logger.info(
          "Training for model ID #{model_id} finished. Remaining epochs less than steps."
        )

        # Perform final merge and reply to the caller
        # This is the state from the last successful merge
        final_model_state = current_model_state

        Hive.Models.Xor.ModelLoader.save_model_state?(final_model_state, "xor.ms")
        Logger.info("Final model state saved for model ID #{model_id}")
        GenServer.reply(Map.fetch!(training_runs, model_id).caller, {:ok, final_model_state})
        # Clean up
        {:noreply, %{state | training_runs: Map.delete(training_runs, model_id)}}
      end
    end
  end

  # This handles the result of a *single* FragmentTrainer task
  @impl true
  def handle_info({ref, {:ok, id, task_model_state}}, state) do
    Logger.info("Received training result for model ID: #{id} from task #{inspect(ref)}")
    %{training_runs: training_runs} = state

    case Map.get(training_runs, id) do
      nil ->
        Logger.warning("Received result for unknown or completed model ID: #{id}")
        {:noreply, state}

      training_run ->
        # Remove the task from active_tasks and add its model_state to completed_model_states
        updated_training_run = %{
          id => %{
            training_run
            | active_tasks:
                Enum.reject(training_run.active_tasks, fn %Task{ref: task_ref} ->
                  task_ref == ref
                end),
              completed_model_states: training_run.completed_model_states ++ [task_model_state]
          }
        }

        updated_training_runs = Map.put(training_runs, id, updated_training_run)
        new_state = %{state | training_runs: updated_training_runs}

        # Check if all tasks for the current step are complete
        if Enum.empty?(updated_training_run.active_tasks) do
          Logger.info(
            "All tasks for step #{updated_training_run.current_step} of model ID #{id} completed. Merging models."
          )

          # Merge the collected model states
          merged_state_for_step =
            merge_models(
              updated_training_run.completed_model_states,
              updated_training_run.current_model_state
            )

          # Update the training run with the new merged state and remaining epochs
          finalized_training_run = %{
            updated_training_run
            | current_model_state: merged_state_for_step,
              remaining_epochs:
                updated_training_run.remaining_epochs - updated_training_run.opts[:steps],
              # Clear completed states for the next step
              completed_model_states: []
          }

          updated_training_runs_after_merge = Map.put(training_runs, id, finalized_training_run)

          # Trigger the next step or finalize training
          if finalized_training_run.remaining_epochs < finalized_training_run.opts[:steps] do
            Logger.info(
              "Training for model ID #{id} complete after step #{finalized_training_run.current_step}."
            )

            # Training is fully complete
            Hive.Models.Xor.ModelLoader.save_model_state?(
              finalized_training_run.current_model_state,
              "xor.ms"
            )

            Logger.info("Final model state saved for model ID #{id}")

            GenServer.reply(
              finalized_training_run.caller,
              {:ok, finalized_training_run.current_model_state}
            )

            # Clean up
            {:noreply,
             %{new_state | training_runs: Map.delete(updated_training_runs_after_merge, id)}}
          else
            # More steps are needed, start the next step
            Logger.info(
              "Starting next training step for model ID #{id}. Remaining epochs: #{finalized_training_run.remaining_epochs}"
            )

            {:noreply, %{new_state | training_runs: updated_training_runs_after_merge},
             {:continue, {:start_step, id}}}
          end
        else
          # Not all tasks for the current step are complete yet
          {:noreply, new_state}
        end
    end
  end

  @impl true
  def handle_info({:DOWN, ref, :process, pid, reason}, state) do
    Logger.warning(
      "Task process #{inspect(pid)}, with reference #{inspect(ref)}, terminated with reason: #{inspect(reason)}"
    )

    # Find which training run this task belonged to and update its status
    updated_training_runs =
      Enum.reduce(state.training_runs, state.training_runs, fn {id, training_run}, acc ->
        if Enum.any?(training_run.active_tasks, fn %Task{ref: task_ref} -> task_ref == ref end) do
          Logger.error(
            "Task for model ID #{id} failed. Reason: #{inspect(reason)}. Consider retrying or marking as failed."
          )

          # Option 1: Remove the failed task and potentially retry
          # Option 2: Mark the training run as failed and notify the caller
          # For simplicity here, just remove the task.
          updated_run = %{
            training_run
            | active_tasks:
                Enum.reject(training_run.active_tasks, fn %Task{ref: task_ref} ->
                  task_ref == ref
                end)
          }

          Map.put(acc, id, updated_run)
        else
          acc
        end
      end)

    {:noreply, %{state | training_runs: updated_training_runs}}
  end

  # --- Utility Functions ---

  defp average(_, a, b) do
    Nx.divide(Nx.add(a, b), 2)
  end

  defp get_next_model_idx_sync(state) do
    state.next_model_id
  end
end
