defmodule Hive.Core.Model do
  @callback build_model() :: %Axon{}
  @callback run_inference(
              model :: %Axon{},
              model_state :: %Axon.ModelState{},
              input_data :: any()
            ) :: any()
end
