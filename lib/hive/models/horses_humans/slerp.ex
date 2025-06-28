defmodule Hive.Models.HorsesHumans.Slerp do
  @dot_threshold 0.9995
  @t 0.4

  def slerp(_, a, b) do
    # Store original shape to reshape later
    original_shape = Nx.shape(a)

    # Flatten tensors to 1D vectors
    flat_a = Nx.reshape(a, {Nx.size(a)})
    flat_b = Nx.reshape(b, {Nx.size(b)})

    # Normalize vectors
    norm_flat_a = Nx.divide(flat_a, Nx.LinAlg.norm(flat_a))
    norm_flat_b = Nx.divide(flat_b, Nx.LinAlg.norm(flat_b))

    # Dot product of normalized vectors
    dot = Nx.sum(Nx.multiply(norm_flat_a, norm_flat_b))

    # If vectors are almost parallel, use linear interpolation
    interpolated_flat =
      if Nx.abs(dot) > @dot_threshold do
        lerp(@t, norm_flat_a, norm_flat_b)
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
        Nx.add(Nx.multiply(y, flat_a), Nx.multiply(z, flat_b))
      end

    # Reshape the result back to the original dimensions
    Nx.reshape(interpolated_flat, original_shape)
  end

  def lerp(t, a, b), do: Nx.add(Nx.multiply(1 - t, a), Nx.multiply(t, b))
end
