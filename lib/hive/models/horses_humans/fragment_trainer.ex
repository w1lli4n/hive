defmodule Hive.Models.HorsesHumans.FragmentTrainer do
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

    optimizer = Polaris.Optimizers.adam(learning_rate: 1.0e-4)
    centralized_optimizer = Polaris.Updates.compose(Polaris.Updates.centralize(), optimizer)

    model_state =
      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, centralized_optimizer)
      |> Axon.Loop.run(data_stream, dematerialized_state,
        epochs: opts[:epochs],
        iterations: opts[:iterations] || 16,
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
