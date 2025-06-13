defmodule Hive.Models.Xor.Model do
  @behaviour Hive.Core.Model

  @impl Hive.Core.Model
  def build_model() do
    a = Axon.input("a", shape: {nil, 1})
    b = Axon.input("b", shape: {nil, 1})

    a
    |> Axon.concatenate(b)
    |> Axon.dense(8, activation: :tanh)
    |> Axon.dense(1, activation: :sigmoid)
  end

  @impl Hive.Core.Model
  def run_inference(model, model_state, input_data) do
    Axon.predict(model, model_state, input_data)
  end
end
