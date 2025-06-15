defmodule Hive.Models.Xor.ModelLoader do
  @behaviour Hive.Core.ModelLoader

  @impl true
  def save_model_state?(model_state, path) do
    # Convert to safe format before serialization
    serializable =
      Nx.serialize(model_state)
      |> :erlang.term_to_binary()

    File.write(path, serializable)
  end

  @impl true
  def load_model_state(path) do
    case File.read(path) do
      {:ok, binary} ->
        model_state = :erlang.binary_to_term(binary)

        loaded_model_state = Nx.deserialize(model_state)

        {:ok, loaded_model_state}

      {:error, reason} ->
        {:error, reason}
    end
  end
end
