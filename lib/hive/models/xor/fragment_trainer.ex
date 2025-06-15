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
end
