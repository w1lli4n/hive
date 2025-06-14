defmodule Hive.Core.ModelLoader do
  @callback load_model_state(model_state_source :: String.t()) ::
              {:ok, %Axon.ModelState{}} | {:error, String.t()}
  @callback save_model_state?(model_state :: %Axon.ModelState{}, path :: String.t()) ::
              :ok | {:error, atom()}
end
