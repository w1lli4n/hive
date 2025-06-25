defmodule Hive.Models.HorsesHumans.DataEgestion do
  @behaviour Hive.Core.DataEgestion

  @impl Hive.Core.DataEgestion
  def process_data(data, :format_tensor) do
    data
    |> Nx.squeeze()
    |> Nx.slice_along_axis(1, 1)
    |> Nx.squeeze()
    |> Nx.to_number()
    |> round()
  end

  @impl Hive.Core.DataEgestion
  def process_data(data, :binary) do
    if data < 1 do
      "horse"
    else
      "human"
    end
  end

  @impl Hive.Core.DataEgestion
  def egest_data(data, :inference) do
    data
    |> process_data(:format_tensor)
    |> process_data(:binary)
  end
end
