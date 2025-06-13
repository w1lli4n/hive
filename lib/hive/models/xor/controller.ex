defmodule Hive.Models.Xor.Controller do
  @behaviour Hive.Core.Controller

  @impl true
  @spec inference_pipeline(input_data :: {integer(), integer()}) ::
          {:ok, integer()} | {:error, String.t()}
  def inference_pipeline(input_data) do
    data =
      input_data
      |> Hive.Models.Xor.DataIngestion.ingest_data(:inference)

    model = Hive.Models.Xor.Model.build_model()

    case Hive.Models.Xor.ModelLoader.load_model_state("xor.ms") do
      {:ok, model_state} ->
        prediction =
          Hive.Models.Xor.Model.run_inference(model, model_state, data)
          |> Hive.Models.Xor.DataEgestion.egest_data(:inference)

        {:ok, prediction}

      {:error, e} ->
        {:error, e}
    end
  end

  @impl true
  def training_pipeline(_data_source) do
    {:ok, loaded_data} =
      Hive.Models.Xor.DataIngestion.load_data("")

    data = loaded_data |> Hive.Models.Xor.DataIngestion.ingest_data(:training)

    model = Hive.Models.Xor.Model.build_model()
    opts = [epochs: 21, steps: 3]

    Hive.Models.Xor.ModelTrainer.run(model, data, opts)
    |> Hive.Models.Xor.ModelLoader.save_model_state?("xor.ms")
  end
end
