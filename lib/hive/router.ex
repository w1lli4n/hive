defmodule Hive.Router do
  use Plug.Router
  require Logger

  plug(:match)
  plug(:dispatch)

  get "/" do
    html_file_path = Path.join(__DIR__, "index.html")

    if File.exists?(html_file_path) do
      case File.read(html_file_path) do
        {:ok, html_content} ->
          conn
          |> put_resp_content_type("text/html")
          |> send_resp(200, html_content)

        {:error, reason} ->
          Logger.error("Failed to read index.html: #{inspect(reason)}")

          conn
          |> put_resp_content_type("application/json")
          |> send_resp(500, Jason.encode!(%{error: "Server error: Could not read HTML file."}))
      end
    else
      Logger.warning("index.html not found at #{html_file_path}")

      conn
      |> put_resp_content_type("application/json")
      |> send_resp(404, Jason.encode!(%{error: "index.html not found"}))
    end
  end

  get "/xor" do
    %{"a" => a, "b" => b} =
      URI.decode_query(conn.query_string)

    data = {a |> String.to_integer(), b |> String.to_integer()}

    case Hive.Models.Xor.Controller.inference_pipeline(data) do
      {:ok, resp} ->
        conn
        |> put_resp_content_type("application/json")
        |> send_resp(200, Jason.encode!(%{result: resp}))

      {:error, e} ->
        conn
        |> put_resp_content_type("application/json")
        |> send_resp(500, Jason.encode!(%{error: e}))
    end
  end

  get "/horses-humans" do
    data = conn.params["data"]

    case Hive.Models.HorsesHumans.Controller.inference_pipeline(data) do
      {:ok, resp} ->
        conn
        |> put_resp_content_type("application/json")
        |> send_resp(200, Jason.encode!(%{result: resp}))

      {:error, e} ->
        conn
        |> put_resp_content_type("application/json")
        |> send_resp(500, Jason.encode!(%{error: e}))
    end
  end

  post "/horses-humans" do
    # Ler o body
    case Plug.Conn.read_body(conn,
           length: 20_000_000,
           read_length: 1_000_000,
           read_timeout: 60_000
         ) do
      {:ok, body, _conn} ->
        case Jason.decode(body) do
          {:ok, %{"data" => data_url}} ->
            case Regex.run(~r/^data:image\/[a-zA-Z]+;base64,(.*)$/, data_url) do
              [_, base64] ->
                case Base.decode64(base64) do
                  {:ok, binary} ->
                    case Hive.Models.HorsesHumans.Controller.inference_pipeline(binary) do
                      {:ok, resp} ->
                        conn
                        |> put_resp_content_type("application/json")
                        |> send_resp(200, Jason.encode!(%{result: resp}))

                      {:error, pipeline_error} ->
                        conn
                        |> put_resp_content_type("application/json")
                        |> send_resp(500, Jason.encode!(%{error: pipeline_error}))
                    end

                  :error ->
                    conn
                    |> put_resp_content_type("application/json")
                    |> send_resp(400, Jason.encode!(%{error: "Invalid Base64 encoding"}))
                end

              nil ->
                conn
                |> put_resp_content_type("application/json")
                |> send_resp(400, Jason.encode!(%{error: "Invalid data URL format"}))
            end

          {:error, decode_error} ->
            conn
            |> put_resp_content_type("application/json")
            |> send_resp(400, Jason.encode!(%{error: "Invalid JSON: #{decode_error}"}))
        end

      {:error, body_error} ->
        conn
        |> put_resp_content_type("application/json")
        |> send_resp(400, Jason.encode!(%{error: "Body read error: #{body_error}"}))
    end
  end

  match _ do
    resp =
      %{error: "Not found"}
      |> Jason.encode!()

    conn
    |> put_resp_content_type("application/json")
    |> send_resp(404, resp)
  end
end
