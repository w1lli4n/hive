defmodule Hive.Router do
  use Plug.Router

  plug(:match)
  plug(:dispatch)

  get "/" do
    resp =
      %{time: DateTime.utc_now()}
      |> Jason.encode!()

    conn
    |> put_resp_content_type("application/json")
    |> send_resp(200, resp)
  end

  get "/xor" do
    a = conn.params["a"] |> String.to_integer()
    b = conn.params["b"] |> String.to_integer()
    data = {a, b}

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

  match _ do
    resp =
      %{error: "Not found"}
      |> Jason.encode!()

    conn
    |> put_resp_content_type("application/json")
    |> send_resp(404, resp)
  end
end
