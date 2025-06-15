defmodule Hive.Models.Xor.FragmentTrainer do
  require Logger
  @behaviour Hive.Core.FragmentTrainer

  @impl Hive.Core.FragmentTrainer
  def run(
        model,
        # This is now the stream of batches
        data_stream,
        opts,
        id,
        initial_model_state \\ Axon.ModelState.empty()
      ) do
    # No unwrapping needed if data_stream is already a Stream yielding batches
    dematerialized_state = Nx.deserialize(initial_model_state)

    model_state =
      model
      |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
      # Pass the stream directly
      |> Axon.Loop.run(data_stream, dematerialized_state,
        epochs: opts[:epochs],
        # Ensure iterations is available
        iterations: opts[:iterations] || 100,
        compiler: EXLA
      )

    materialized_state = Nx.serialize(model_state)
    {:ok, id, materialized_state}
  rescue
    e ->
      Logger.error("FragmentTrainer failed for model ID #{id}: #{inspect(e)}")
      {:error, :training_failed}
  end

  def materialize_model_state(%Axon.ModelState{parameters: params, state: state} = model_state) do
    materialized_params = materialize_map_tensors(params)
    materialized_state = materialize_map_tensors(state)

    %Axon.ModelState{model_state | parameters: materialized_params, state: materialized_state}
  end

  # Handles nested maps (e.g., for different layers' parameters/state)
  defp materialize_map_tensors(map) when is_map(map) do
    Enum.into(map, %{}, fn {key, value} ->
      {key, materialize_tensor_or_map(value)}
    end)
  end

  # Handles lists (e.g., if you have lists of tensors in your state)
  defp materialize_map_tensors(list) when is_list(list) do
    Enum.map(list, &materialize_tensor_or_map/1)
  end

  # Base case: Materialize an Nx.Tensor
  defp materialize_tensor_or_map(tensor) when is_struct(tensor, Nx.Tensor) do
    {:nx_binary, Nx.to_binary(tensor), Nx.type(tensor), Nx.shape(tensor)}
  end

  # Recursive case for nested maps or lists
  defp materialize_tensor_or_map(value) when is_map(value) or is_list(value) do
    materialize_map_tensors(value)
  end

  # Fallback for any other data type (e.g., scalars, strings) - leave as is
  defp materialize_tensor_or_map(other) do
    other
  end

  def dematerialize_model_state(%Axon.ModelState{parameters: params, state: state} = model_state) do
    dematerialized_params = dematerialize_map_tensors(params)
    dematerialized_state = dematerialize_map_tensors(state)

    %Axon.ModelState{model_state | parameters: dematerialized_params, state: dematerialized_state}
  end

  defp dematerialize_map_tensors(map) when is_map(map) do
    Enum.into(map, %{}, fn {key, value} ->
      {key, dematerialize_binary_or_map(value)}
    end)
  end

  defp dematerialize_map_tensors(list) when is_list(list) do
    Enum.map(list, &dematerialize_binary_or_map/1)
  end

  defp dematerialize_binary_or_map({:nx_binary, binary, type, shape})
       when is_binary(binary) and is_atom(type) and is_list(shape) do
    # This is the correct way to reconstruct with Nx.from_binary/3
    Nx.from_binary(binary, type, shape)
  end

  # Keep these existing clauses, but consider the warning in the 'other' clause
  defp dematerialize_binary_or_map(value) when is_map(value) or is_list(value) do
    dematerialize_map_tensors(value)
  end

  defp dematerialize_binary_or_map(other) do
    # This will now only catch things that are NOT {:nx_binary, ...} tuples
    # and were not handled by map/list recursion.
    # The error was likely due to this receiving something unexpected previously.
    # IO.warn("Dematerialize: Encountered non-Nx binary or unexpected data: #{inspect(other)}")
    other
  end
end
