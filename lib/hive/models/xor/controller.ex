defmodule Hive.Models.Xor.Controller do
  @behaviour Hive.Core.Controller

  require Logger

  @impl true
  @spec inference_pipeline(input_data :: {integer(), integer()}) ::
          {:ok, integer()} | {:error, String.t()}
  def inference_pipeline(input_data) do
    data =
      input_data
      |> Hive.Models.Xor.DataIngestion.ingest_data(:inference)

    model = Hive.Models.Xor.Model.build_model()

    case Hive.Models.Xor.ModelLoader.load_model_state("models/xor.ms") do
      {:ok, model_state} ->
        Nx.global_default_backend(EXLA.Backend)

        prediction =
          model
          |> Hive.Models.Xor.Model.run_inference(model_state, data)
          |> Hive.Models.Xor.DataEgestion.egest_data(:inference)

        {:ok, prediction}

      {:error, e} ->
        {:error, e}
    end
  end

  @impl true
  def training_pipeline(_data_source) do
    data_stream = Hive.Models.Xor.DataIngestion.ingest_data(nil, :training)

    model = Hive.Models.Xor.Model.build_model()
    opts = [epochs: 10, steps: 1, iterations: 1000]
    {:ok, _model_state} = Hive.Models.Xor.ModelTrainer.run(model, data_stream, opts)

    Logger.info("Training completed")
  end
end
