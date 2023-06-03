package org.example.neural;

public class CrossEntropy implements SimpleNeuralNetwork.Loss {
    @Override
    public double loss(Vector v, Vector y) throws Exception {
        for(int i=0;i<y.size();i++) {
            if(y.x(i) == 1) {
                return -Math.log(v.x(i) + 0.00001f);
            }
        }

        throw new Exception("Label has some problem with it!");
    }

    @Override
    public Vector derivativeByA(Vector A, Vector Y) {
        Vector derivative = new Vector(A.size());

        double deri = 0;
        for(int i=0;i<Y.size();i++) {
            if(Y.x(i) == 1) {
                deri = -1 / (A.x(i) + 1e-15f);
            }
        }

        for(int i=0;i<Y.size();i++) {
            derivative.setX(i, deri);
        }

        return derivative;
    }
}