defmodule Hive.Models.HorsesHumans.Model do
  @behaviour Hive.Core.Model

  @impl Hive.Core.Model
  def build_model() do
    Axon.input("input", shape: {nil, 300, 300, 4})
    |> Axon.conv(16, kernel_size: {3, 3}, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(32, kernel_size: {3, 3}, activation: :relu)
    |> Axon.spatial_dropout(rate: 0.5)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
    |> Axon.spatial_dropout(rate: 0.5)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.flatten()
    |> Axon.dropout(rate: 0.5)
    |> Axon.dense(512, activation: :relu)
    |> Axon.dense(2, activation: :softmax)
  end

  @impl Hive.Core.Model
  def run_inference(model, model_state, input_data) do
    Axon.predict(model, model_state, input_data)
  end
end
