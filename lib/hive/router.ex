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
    data = conn.params["data"]

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

  match _ do
    resp =
      %{error: "Not found"}
      |> Jason.encode!()

    conn
    |> put_resp_content_type("application/json")
    |> send_resp(404, resp)
  end
end
