defmodule Hive.MixProject do
  use Mix.Project

  def project do
    [
      app: :hive,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {Hive.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
      {:plug, "~> 1.18"},
      {:bandit, "~> 1.6.11"},
      {:jason, "~> 1.4.4"},
      {:axon, "~> 0.7.0"},
      {:nx, "~> 0.9.2"},
      {:exla, "~> 0.9.2"},
      {:libcluster, "~> 3.5.0"}
    ]
  end
end
