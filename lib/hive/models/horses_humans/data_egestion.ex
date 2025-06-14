defmodule Hive.Models.HorsesHumans.DataEgestion do
  @behaviour Hive.Core.DataEgestion

  @impl Hive.Core.DataEgestion
  def process_data(data, :format_tensor) do
    data
    |> Nx.round()
    |> Nx.squeeze()
    |> Nx.to_number()
    |> round()
  end

  @impl Hive.Core.DataEgestion
  def egest_data(data, :inference) do
    data
    |> process_data(:format_tensor)
  end
end
