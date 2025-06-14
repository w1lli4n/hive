defmodule Hive.Models.HorsesHumans.ModelLoader do
  @behaviour Hive.Core.ModelLoader

  @impl true
  def save_model_state?(model_state, path) do
    # Convert to safe format before serialization
    serializable =
      Axon.ModelState.freeze(model_state)
      |> Axon.ModelState.frozen_parameters()
      |> Enum.map(fn {section_name, section_data} ->
        converted_section_data =
          Enum.map(section_data, fn {key, tensor} ->
            # Convert the tensor to a serializable form
            {key, Nx.to_list(tensor)}
          end)
          |> Map.new()

        {section_name, converted_section_data}
      end)
      |> Map.new()
      |> :erlang.term_to_binary()

    File.write(path, serializable)
  end

  @impl true
  def load_model_state(path) do
    case File.read(path) do
      {:ok, binary} ->
        ms = :erlang.binary_to_term(binary)

        lms =
          Enum.map(ms, fn {section_name, section_data} ->
            transferred_section_data =
              Enum.map(section_data, fn {key, tensor} ->
                {key, Nx.tensor(tensor)}
              end)
              |> Map.new()

            {section_name, transferred_section_data}
          end)
          |> Map.new()

        {:ok, lms}

      {:error, reason} ->
        {:error, reason}
    end
  end
end
