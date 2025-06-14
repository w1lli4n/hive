defmodule Hive.Models.HorsesHumans.DataIngestion do
  import Nx.Defn
  @behaviour Hive.Core.DataIngestion

  @batch_size 32

  @impl Hive.Core.DataIngestion
  def load_data(data_source) do
    data_stream =
      Path.join([data_source, "**", "*.png"])
      |> Path.wildcard()
      |> Stream.map(fn path ->
        {Path.relative_to(path, data_source), File.read!(path)}
      end)

    {:ok, data_stream}
  end

  @impl Hive.Core.DataIngestion
  def process_data({filename, binary}, :parse_file) do
    label =
      if String.starts_with?(filename, "horses/"),
        do: Nx.tensor([1, 0], type: {:u, 8}),
        else: Nx.tensor([0, 1], type: {:u, 8})

    image = binary |> StbImage.read_binary!() |> StbImage.to_nx()

    {image, label}
  end

  # @impl Hive.Core.DataIngestion
  # def process_data(data, :inference) do
  # end

  @impl Hive.Core.DataIngestion
  def ingest_data(files, :training) do
    files
    |> Enum.shuffle()
    |> Stream.chunk_every(@batch_size, @batch_size, :discard)
    |> Task.async_stream(
      fn batch ->
        {images, labels} =
          batch
          |> Enum.map(fn {filename, binary} -> process_data({filename, binary}, :parse_file) end)
          |> Enum.unzip()

        {Nx.stack(images), Nx.stack(labels)}
      end,
      timeout: :infinity
    )
    |> Stream.map(fn {:ok, {images, labels}} -> {augment(images), labels} end)
    |> Stream.cycle()
  end

  # @impl Hive.Core.DataIngestion
  # def ingest_data(data, :inference) do
  # data
  # |> process_data(:inference)
  # end

  defnp augment(images) do
    # Normalize
    images = images / 255.0

    # Optional vertical/horizontal flip
    {u, _new_key} = Nx.Random.key(1987) |> Nx.Random.uniform()

    cond do
      u < 0.25 -> images
      u < 0.5 -> Nx.reverse(images, axes: [2])
      u < 0.75 -> Nx.reverse(images, axes: [3])
      true -> Nx.reverse(images, axes: [2, 3])
    end
  end
end
