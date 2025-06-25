defmodule Hive.Models.Xor.DataIngestion do
  @behaviour Hive.Core.DataIngestion

  @batch_size 32

  @impl Hive.Core.DataIngestion
  def load_data(_data_source) do
    a = Nx.tensor(for _ <- 1..@batch_size, do: [Enum.random(0..1)])
    b = Nx.tensor(for _ <- 1..@batch_size, do: [Enum.random(0..1)])
    y = Nx.logical_xor(a, b)
    # <--- Return the single batch tuple directly
    {:ok, {%{"a" => a, "b" => b}, y}}
  end

  @impl Hive.Core.DataIngestion
  def process_data(_data, :training) do
    Stream.repeatedly(fn ->
      {:ok, batch} = load_data("")
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
  def ingest_data(_data, :training) do
    process_data(nil, :training)
  end

  @impl Hive.Core.DataIngestion
  def ingest_data(data, :inference) do
    data
    |> process_data(:inference)
  end
end
