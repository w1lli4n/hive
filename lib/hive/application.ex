defmodule Hive.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      {Bandit, plug: Hive.Router, scheme: :http, port: 3177},
      {Cluster.Supervisor, [Application.get_env(:libcluster, :topologies)]},
      {Task.Supervisor, name: Hive.TaskSupervisor},
      Hive.Models.Xor,
      Hive.Models.HorsesHumans
    ]

    opts = [strategy: :one_for_one, name: Hive.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
