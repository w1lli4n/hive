defmodule Hive.Models.HorsesHumans.Slerp do
  @dot_threshold 0.9995
  @t 0.4

  def slerp(_, a, b) do
    # Normalize vectors
    norm_a = Nx.divide(a, Nx.LinAlg.norm(a))
    norm_b = Nx.divide(b, Nx.LinAlg.norm(b))

    # Dot product of normalized vectors
    dot = Nx.sum(Nx.multiply(norm_a, norm_b))

    # If vectors are almost parallel, use linear interpolation
    if Nx.abs(dot) > @dot_threshold do
      lerp(@t, norm_a, norm_b)
    else
      # Calculate angle between vectors
      theta_0 = Nx.acos(dot)
      sin_theta_0 = Nx.sin(theta_0)

      # Calculate intermediate angle
      theta_t = Nx.multiply(theta_0, @t)
      sin_theta_t = Nx.sin(theta_t)

      # Calculate coefficients
      y = Nx.divide(Nx.sin(Nx.subtract(theta_0, theta_t)), sin_theta_0)
      z = Nx.divide(sin_theta_t, sin_theta_0)

      # Interpolate vectors
      Nx.add(Nx.multiply(y, a), Nx.multiply(z, b))
    end
  end

  def lerp(t, a, b), do: Nx.add(Nx.multiply(1 - t, a), Nx.multiply(t, b))
end
