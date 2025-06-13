defmodule Hive.Core.DataIngestion do
  @callback load_data(data_source :: String.t()) ::
              {:ok, Enumerable.t()} | {:error, String.t()}
  @callback process_data(data :: Enumerable.t(), flag :: atom()) :: Enumerable.t()
  @callback ingest_data(data :: Enumerable.t(), flag :: atom()) :: Enumerable.t()
end
