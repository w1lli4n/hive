defmodule Hive.Core.ModelTrainer do
  @callback run(
              model :: %Axon{},
              data :: {Enumerable.t(), Enumerable.t()},
              opts :: Enumerable.t(),
              id :: integer()
            ) :: {:ok, %Axon{}} | {:error, String.t()}

  @callback merge_models(
              model_states :: [%Axon.ModelState{}],
              initial_model_state :: %Axon.ModelState{}
            ) :: %Axon{}
end
