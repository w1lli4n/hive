defmodule Hive.Models.Xor.ModelTrainer do
  @behaviour Hive.Core.ModelTrainer
  use GenServer

  require Logger

  # --- API ---
  def start_link(_opts) do
    GenServer.start_link(__MODULE__, :ok, name: __MODULE__)
  end

  @impl Hive.Core.ModelTrainer
  def run(model, data, opts, id \\ nil, initial_model_state \\ Axon.ModelState.empty()) do
    GenServer.call(
      __MODULE__,
      {:start_full_training, model, data, opts, id, initial_model_state},
      :infinity
    )
  end

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
    {:ok, %{nodes: Node.list() ++ [Node.self()], training_runs: %{}, next_model_id: 0}}
  end

  @impl true
  def handle_call(
        {:start_full_training, model, data, opts, model_id, initial_model_state},
        from,
        state
      ) do
    generated_id = model_id || state.next_model_id
    id_str = to_string(generated_id)

    training_run_state = %{
      caller: from,
      model: model,
      data: data,
      opts: opts,
      current_model_state: initial_model_state,
      remaining_epochs: opts[:epochs],
      active_tasks: [],
      completed_model_states: [],
      current_step: 0
    }

    updated_training_runs = Map.put(state.training_runs, id_str, training_run_state)

    new_state = %{
      state
      | training_runs: updated_training_runs,
        next_model_id: generated_id + 1
    }

    {:noreply, new_state, {:continue, {:start_step, id_str}}}
  end

  @impl true
  def handle_call(:get_next_model_id_sync, _from, state) do
    {:reply, {:ok, state.next_model_id}, %{state | next_model_id: state.next_model_id + 1}}
  end

  @impl true
  def handle_continue({:start_step, model_id}, state) do
    case Map.get(state.training_runs, model_id) do
      nil ->
        Logger.error("Training run #{model_id} not found")
        {:noreply, state}

      training_run ->
        %{nodes: nodes} = state

        %{
          model: _model,
          data: _data,
          opts: opts,
          current_model_state: _current_model_state,
          remaining_epochs: remaining_epochs
        } = training_run

        if Enum.empty?(nodes) do
          Logger.warning("No nodes available for training for model ID #{model_id}.")
          GenServer.reply(training_run.caller, {:error, "no_nodes_available"})
          {:noreply, %{state | training_runs: Map.delete(state.training_runs, model_id)}}
        else
          if remaining_epochs >= opts[:steps] do
            start_training_step(model_id, training_run, nodes, state)
          else
            complete_training(model_id, training_run, state)
          end
        end
    end
  end

  @impl true
  def handle_info({ref, {:ok, id, task_model_state}}, state) do
    id_str = to_string(id)
    Logger.info("Received training result for model ID: #{id_str} from task #{inspect(ref)}")

    case Map.get(state.training_runs, id_str) do
      nil ->
        Logger.warning("Received result for unknown or completed model ID: #{id_str}")
        {:noreply, state}

      training_run ->
        updated_run =
          training_run
          |> remove_completed_task(ref)
          |> add_completed_model_state(task_model_state)

        updated_runs = Map.put(state.training_runs, id_str, updated_run)

        if all_tasks_completed?(updated_run) do
          handle_step_completion(id_str, updated_run, updated_runs, state)
        else
          {:noreply, %{state | training_runs: updated_runs}}
        end
    end
  end

  @impl true
  def handle_info({:DOWN, ref, :process, pid, reason}, state) do
    Logger.warning(
      "Task process #{inspect(pid)}, with reference #{inspect(ref)}, terminated with reason: #{inspect(reason)}"
    )

    updated_runs =
      Enum.reduce(state.training_runs, state.training_runs, fn {id, run}, acc ->
        if Enum.any?(run.active_tasks, fn %Task{ref: task_ref} -> task_ref == ref end) do
          Logger.error("Task for model ID #{id} failed. Reason: #{inspect(reason)}")
          updated_run = %{run | active_tasks: Enum.reject(run.active_tasks, &(&1.ref == ref))}
          Map.put(acc, id, updated_run)
        else
          acc
        end
      end)

    {:noreply, %{state | training_runs: updated_runs}}
  end

  @impl true
  def handle_info({:nodeup, node}, state) do
    Logger.info("Node #{node} is up")
    {:noreply, state}
  end

  @impl true
  def handle_info({:nodedown, node}, state) do
    Logger.info("Node #{node} is down")
    {:noreply, state}
  end

  # --- Private Functions ---
  defp average(_, a, b), do: Nx.divide(Nx.add(a, b), 2)

  defp start_training_step(model_id, training_run, nodes, state) do
    %{
      model: model,
      data: data,
      opts: opts,
      current_model_state: current_model_state
    } = training_run

    Logger.info(
      "Starting training step for model ID: #{model_id}, remaining epochs: #{training_run.remaining_epochs}"
    )

    current_step_tasks =
      Enum.map(nodes, fn node ->
        Task.Supervisor.async_nolink(
          {Hive.TaskSupervisor, node},
          Hive.Models.Xor.FragmentTrainer,
          :run,
          [model, data, opts, model_id, current_model_state]
        )
      end)

    updated_run = %{
      training_run
      | active_tasks: current_step_tasks,
        completed_model_states: [],
        current_step: training_run.current_step + 1
    }

    updated_runs = Map.put(state.training_runs, model_id, updated_run)
    {:noreply, %{state | training_runs: updated_runs}}
  end

  defp remove_completed_task(training_run, ref) do
    %{training_run | active_tasks: Enum.reject(training_run.active_tasks, &(&1.ref == ref))}
  end

  defp add_completed_model_state(training_run, model_state) do
    %{training_run | completed_model_states: training_run.completed_model_states ++ [model_state]}
  end

  defp all_tasks_completed?(%{active_tasks: active_tasks}), do: Enum.empty?(active_tasks)

  defp handle_step_completion(id, training_run, training_runs, state) do
    Logger.info(
      "All tasks for step #{training_run.current_step} of model ID #{id} completed. Merging models."
    )

    merged_state =
      merge_models(
        training_run.completed_model_states,
        training_run.current_model_state
      )

    updated_run = %{
      training_run
      | current_model_state: merged_state,
        remaining_epochs: training_run.remaining_epochs - training_run.opts[:steps],
        completed_model_states: []
    }

    updated_runs = Map.put(training_runs, id, updated_run)

    if training_complete?(updated_run) do
      complete_training(id, updated_run, updated_runs, state)
    else
      Logger.info(
        "Starting next training step for model ID #{id}. Remaining epochs: #{updated_run.remaining_epochs}"
      )

      {:noreply, %{state | training_runs: updated_runs}, {:continue, {:start_step, id}}}
    end
  end

  defp training_complete?(%{remaining_epochs: remaining, opts: opts}) do
    remaining < opts[:steps]
  end

  defp complete_training(id, training_run, state) do
    complete_training(id, training_run, state.training_runs, state)
  end

  defp complete_training(id, training_run, training_runs, state) do
    Logger.info("Training for model ID #{id} complete after step #{training_run.current_step}.")

    Hive.Models.Xor.ModelLoader.save_model_state?(
      training_run.current_model_state,
      "/models/xor.ms"
    )

    Logger.info("Final model state saved for model ID #{id}")
    GenServer.reply(training_run.caller, {:ok, training_run.current_model_state})
    {:noreply, %{state | training_runs: Map.delete(training_runs, id)}}
  end
end
