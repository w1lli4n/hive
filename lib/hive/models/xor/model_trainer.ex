defmodule Hive.Models.Xor.ModelTrainer do
  @behaviour Hive.Core.ModelTrainer
  use GenServer

  require Logger

  # --- API ---

  @doc """
  Starts the GenServer for the ModelTrainer.
  """
  def start_link(_opts) do
    GenServer.start_link(__MODULE__, :ok, name: __MODULE__)
  end

  @doc """
  Runs the model training process across available nodes.
  """
  @impl Hive.Core.ModelTrainer
  def run(model, data, opts, id \\ nil, initial_model_state \\ Axon.ModelState.empty()) do
    if opts[:epochs] < opts[:steps] do
      Hive.Models.Xor.ModelLoader.save_model_state?(initial_model_state, "xor.ms")
      Logger.info(IO.inspect(initial_model_state))
      {:ok, initial_model_state}
    else
      generated_id = id || get_next_model_idx()

      case GenServer.call(
             __MODULE__,
             {:run_training, model, data, opts, generated_id, initial_model_state}
           ) do
        {:error, :no_nodes_available} ->
          {:error, "no_nodes_available"}

        {:ok, :tasks_running} ->
          run(model, data, opts, id, initial_model_state)

        {:ok, current_id, current_model_state} ->
          new_data = data
          new_epochs = opts[:epochs] - opts[:steps]
          new_opts = Keyword.put(opts, :epochs, new_epochs)

          run(model, new_data, new_opts, current_id, current_model_state)
      end
    end
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
    # Monitor node changes to update the list of available nodes
    :net_kernel.monitor_nodes(true)

    # Initialize state with self node and an empty map for models
    {:ok, %{nodes: Node.list() ++ [Node.self()], models: %{}, next_model_id: 0}}
  end

  @impl true
  def handle_call({:run_training, model, data, opts, model_id, initial_model_state}, _from, state) do
    %{nodes: nodes, models: models} = state

    if Enum.empty?(nodes) do
      Logger.warning("No nodes available for training.")
      {:reply, {:error, :no_nodes_available}, state}
    end

    # Check if there are active tasks for this model_id

    tasks =
      case Map.get(models, model_id) do
        nil -> []
        tasks -> tasks
      end

    if Enum.empty?(tasks) do
      Logger.info("Tasks are already running for model ID: #{model_id}")
      {:reply, {:ok, :tasks_running}, state}
    end

    # Initialize or retrieve model state
    current_model_states =
      case Map.get(models, model_id) do
        %{model_states: ms} -> ms
        _ -> []
      end

    # Merge existing model states with the initial_model_state
    merged_initial_state =
      merge_models(current_model_states, initial_model_state)

    # Prepare the model entry in the state map
    updated_models =
      Map.put(models, model_id, %{
        id: model_id,
        active_tasks: [],
        # Reset model_states for the new training round
        model_states: []
      })

    # Distribute tasks to nodes
    new_state =
      Enum.reduce(nodes, %{state | models: updated_models}, fn node, acc_state ->
        task =
          {Hive.TaskSupervisor, node}
          |> Task.Supervisor.async_nolink(
            Hive.Models.Xor.FragmentTrainer,
            :run,
            [model, data, opts, model_id, merged_initial_state]
          )

        # Add the task to the model's active_tasks list
        %{acc_state | models: add_task_to_model(acc_state.models, model_id, task)}
      end)

    {:reply, {:ok, model_id, merged_initial_state}, new_state}
  end

  @impl true
  def handle_call(:get_next_model_id, _from, state) do
    %{next_model_id: next_model_id} = state
    {:reply, {:ok, next_model_id}, %{state | next_model_id: next_model_id + 1}}
  end

  @impl true
  def handle_info({ref, {:ok, id, model_state}}, state) do
    Logger.info("Received training result for model ID: #{id}")

    updated_models =
      state.models
      |> remove_task_from_model(id, ref)
      |> add_model_state_to_model(id, model_state)

    {:noreply, %{state | models: updated_models}}
  end

  @impl true
  def handle_info({:DOWN, ref, :process, pid, reason}, state) do
    Logger.warning(
      "Task process #{inspect(pid)}, with reference #{inspect(ref)}, terminated with reason: #{inspect(reason)}"
    )

    # Potentially update state to reflect the downed task, e.g., remove from active_tasks
    {:noreply, state}
  end

  # --- Utility Functions ---

  defp average(_, a, b) do
    Nx.divide(Nx.add(a, b), 2)
  end

  # Helper function to add a task to a specific model's active_tasks list
  defp add_task_to_model(models, model_id, task) do
    Map.update(
      models,
      model_id,
      # Default if model_id not found (shouldn't happen with current logic)
      %{},
      fn model_entry ->
        %{model_entry | active_tasks: model_entry.active_tasks ++ [task]}
      end
    )
  end

  # Helper function to remove a task from a specific model's active_tasks list
  defp remove_task_from_model(models, model_id, ref) do
    Map.update(
      models,
      model_id,
      %{},
      fn model_entry ->
        %{
          model_entry
          | active_tasks:
              Enum.reject(model_entry.active_tasks, fn %Task{ref: task_ref} -> task_ref == ref end)
        }
      end
    )
  end

  # Helper function to add a model state to a specific model's model_states list
  defp add_model_state_to_model(models, model_id, model_state) do
    Map.update(
      models,
      model_id,
      %{},
      fn model_entry ->
        %{model_entry | model_states: model_entry.model_states ++ [model_state]}
      end
    )
  end

  defp get_next_model_idx do
    {:ok, id} = GenServer.call(__MODULE__, :get_next_model_id)
    id
  end
end
