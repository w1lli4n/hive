defmodule Hive.Core.DataEgestion do
  @callback process_data(data :: any(), flag :: atom()) :: any()
  @callback egest_data(data :: any(), flag :: atom()) :: any()
end
