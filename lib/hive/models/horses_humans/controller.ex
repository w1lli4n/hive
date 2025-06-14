defmodule Hive.Models.HorsesHumans.Controller do
  @behaviour Hive.Core.Controller

  require Logger

  @impl true
  @spec inference_pipeline(input_data :: {integer(), integer()}) ::
          {:ok, integer()} | {:error, String.t()}
  def inference_pipeline(input_data) do
    data =
      input_data
      |> Hive.Models.HorsesHumans.DataIngestion.ingest_data(:inference)

    model = Hive.Models.HorsesHumans.Model.build_model()

    case Hive.Models.HorsesHumans.ModelLoader.load_model_state("horses_humans.ms") do
      {:ok, model_state} ->
        # Ensure we're using the same backend as during training
        Nx.global_default_backend(EXLA.Backend)

        prediction =
          model
          |> Hive.Models.HorsesHumans.Model.run_inference(model_state, data)
          |> Hive.Models.HorsesHumans.DataEgestion.egest_data(:inference)

        {:ok, prediction}

      {:error, e} ->
        {:error, e}
    end
  end

  @impl true
  def training_pipeline(_data_source) do
    # Here, instead of loading one batch and then streaming it,
    # we directly get the infinite stream of batches for training.
    url = "https://storage.googleapis.com/learning-datasets/horse-or-human.zip"

    {:ok, data} =
      Hive.Models.HorsesHumans.DataIngestion.load_data(url)

    data_stream = data |> Hive.Models.HorsesHumans.DataIngestion.ingest_data(:training)

    model = Hive.Models.HorsesHumans.Model.build_model()
    opts = [epochs: 20, steps: 5, iterations: 16]

    # Pass the stream
    {:ok, _model_state} = Hive.Models.HorsesHumans.ModelTrainer.run(model, data_stream, opts)

    Logger.info("Training completed")
  end
end
