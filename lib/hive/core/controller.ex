defmodule Hive.Core.Controller do
  @callback inference_pipeline(input_data :: any) :: {:ok, any} | {:error, String.t()}
  @callback training_pipeline(data_source :: String.t()) :: {:ok, any} | {:error, String.t()}
end
