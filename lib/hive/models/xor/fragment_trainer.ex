defmodule Hive.Models.Xor.FragmentTrainer do
  @behaviour Hive.Core.FragmentTrainer

  @impl Hive.Core.FragmentTrainer
  def run(
        model,
        data,
        opts,
        id,
        initial_model_state \\ Axon.ModelState.empty()
      ) do
    model_state =
      model
      |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
      |> Axon.Loop.run(data, initial_model_state,
        epochs: opts[:epochs],
        iterations: 100,
        compiler: EXLA
      )

    {:ok, id, model_state}
  end
end
