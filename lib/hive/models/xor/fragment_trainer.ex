defmodule Hive.Models.Xor.FragmentTrainer do
  require Logger
  @behaviour Hive.Core.FragmentTrainer

  @impl Hive.Core.FragmentTrainer
  def run(
        model,
        data_stream,
        opts,
        id,
        initial_model_state \\ Axon.ModelState.empty()
      ) do
    dematerialized_state = Nx.deserialize(initial_model_state)

    model_state =
      model
      |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
      |> Axon.Loop.metric(:accuracy)
      |> Axon.Loop.run(data_stream, dematerialized_state,
        epochs: opts[:epochs],
        iterations: opts[:iterations] || 1000,
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
