package org.example.neural;

public class LinearActivation extends ActivationFunction {
    LinearActivation(int featureSize, int neurons) {
        super(featureSize, neurons);
    }

    @Override
    public Vector out(Vector z) {
        return z;
    }

    @Override
    public Vector derivativeByZ(Vector z, Vector y) {
        return new Vector(z.size(), 1.0f);
    }

}