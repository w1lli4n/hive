defmodule Hive.Core.FragmentTrainer do
  @callback run(
              model :: %Axon{},
              data :: {Enumerable.t(), Enumerable.t()},
              initial_model_state :: %Axon.ModelState{},
              opts :: Enumerable.t()
            ) :: {:ok, integer(), any} | {:error, String.t()}
end
