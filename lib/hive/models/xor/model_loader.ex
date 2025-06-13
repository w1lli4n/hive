defmodule Hive.Models.Xor.ModelLoader do
  @behaviour Hive.Core.ModelLoader

  @impl Hive.Core.ModelLoader
  def load_model(model_source) do
    case File.read(model_source) do
      {:ok, binary} -> {:ok, :erlang.binary_to_term(binary)}
      {:error, reason} -> {:error, reason}
    end
  end

  @impl Hive.Core.ModelLoader
  def load_model_state(model_state_source) do
    case File.read(model_state_source) do
      {:ok, binary} -> {:ok, :erlang.binary_to_term(binary)}
      {:error, reason} -> {:error, reason}
    end
  end

  @impl Hive.Core.ModelLoader
  def save_model?(model, path) do
    binary = :erlang.term_to_binary(model)

    case File.write(path, binary) do
      :ok -> {:ok, model}
      {:error, reason} -> {:error, reason}
    end
  end

  @impl Hive.Core.ModelLoader
  def save_model_state?(model_state, path) do
    binary = :erlang.term_to_binary(model_state)

    case File.write(path, binary) do
      :ok -> {:ok, model_state}
      {:error, reason} -> {:error, reason}
    end
  end
end
