defmodule Hive.Core.ModelLoader do
  @callback load_model(model_source :: String.t()) :: {:ok, %Axon{}} | {:error, String.t()}
  @callback load_model_state(model_state_source :: String.t()) ::
              {:ok, %Axon.ModelState{}} | {:error, String.t()}
  @callback save_model?(model :: %Axon{}, path :: String.t()) ::
              {:ok, %Axon{}} | {:error, String.t()}
  @callback save_model_state?(model_state :: %Axon.ModelState{}, path :: String.t()) ::
              {:ok, %Axon.ModelState{}} | {:error, String.t()}
end
