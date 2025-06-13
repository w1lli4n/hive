defmodule Hive.Models.Xor.DataIngestion do
  @behaviour Hive.Core.DataIngestion

  @batch_size 32

  # load_data should probably generate a single batch, or be renamed
  # if its purpose is to create the data *for* a single batch.
  # Let's make load_data generate a single batch tuple for clarity.
  @impl Hive.Core.DataIngestion
  def load_data(_data_source) do
    a = Nx.tensor(for _ <- 1..@batch_size, do: [Enum.random(0..1)])
    b = Nx.tensor(for _ <- 1..@batch_size, do: [Enum.random(0..1)])
    y = Nx.logical_xor(a, b)
    # <--- Return the single batch tuple directly
    {:ok, {%{"a" => a, "b" => b}, y}}
  end

  # This function will create a stream of batches for training.
  # It will repeatedly call `load_data` to generate new batches.
  @impl Hive.Core.DataIngestion
  def process_data(_data, :training) do
    # Here, 'data' is the result of 'load_data' (a single batch tuple)
    # We want a stream of *newly generated* batches.
    # So, we'll repeatedly call load_data without arguments to generate new ones.
    Stream.repeatedly(fn ->
      # Call load_data to get a new batch tuple
      {:ok, batch} = load_data("")
      # Return just the batch tuple
      batch
    end)
  end

  @impl Hive.Core.DataIngestion
  def process_data(data, :inference) do
    {a, b} = data
    x1 = Nx.tensor([[a]])
    x2 = Nx.tensor([[b]])

    %{"a" => x1, "b" => x2}
  end

  @impl Hive.Core.DataIngestion
  # _data is unused now, as we generate new batches
  def ingest_data(_data, :training) do
    # Pass nil or anything as data, it's ignored now
    process_data(nil, :training)
  end

  @impl Hive.Core.DataIngestion
  def ingest_data(data, :inference) do
    data
    |> process_data(:inference)
  end
end
