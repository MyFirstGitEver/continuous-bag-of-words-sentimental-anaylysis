package org.example.neural;

import java.util.Arrays;
import java.util.Random;

public class Vector {

    private double[] points;

    public Vector(double... points) {
        this.points = points;
    }

    public Vector(int size) {
        points = new double[size];
    }

    public Vector(int size, double value) {
        points = new double[size];

        Arrays.fill(points, value);
    }

    public Vector(Object[] data) {
        points = new double[data.length];

        for (int i = 0; i < points.length; i++) {
            points[i] = (Double) data[i];
        }
    }

    public Vector(String data, String regex) {
        String[] list = data.split(regex);

        points = new double[list.length];

        for (int i = 0; i < points.length; i++) {
            points[i] = Double.parseDouble(list[i]);
        }
    }

    public Vector(String data) {
        String[] list = data.split("\t");

        points = new double[list.length];

        for (int i = 0; i < points.length; i++) {
            points[i] = Double.parseDouble(list[i]);
        }
    }

    public Vector(Vector[] twoD) throws Exception {
        // flatten this 2d matrix

        if (twoD.length != 1 && twoD[0].size() != 1) {
            throw new Exception("Can't flatten this one!");
        } else if (twoD.length == 1) {
            points = new double[twoD[0].size()];

            for (int i = 0; i < twoD[0].size(); i++) {
                points[i] = twoD[0].x(i);
            }
        } else {
            points = new double[twoD.length];

            for (int j = 0; j < twoD.length; j++) {
                points[j] = twoD[j].x(0);
            }
        }
    }

    public double x(int i) {
        return points[i];
    }

    public void setX(int pos, double value) {
        points[pos] = value;
    }

    public int size() {
        return points.length;
    }

    public double dot(Vector w) {
        if (points.length != w.size()) {
            return Double.NaN;
        }

        int n = points.length;
        double total = 0;
        for (int i = 0; i < n; i++) {
            total += points[i] * w.x(i);
        }

        return total;
    }

    public void subtract(Vector v) {
        for (int i = 0; i < points.length; i++) {
            points[i] -= v.x(i);
        }
    }

    public Vector scaleBy(double x) {
        for (int i = 0; i < points.length; i++) {
            points[i] *= x;
        }

        return this;
    }

    public Vector copy() {
        return new Vector(Arrays.copyOfRange(points, 0, points.length));
    }

    public double sum() {
        double total = 0;

        for (double point : points) {
            total += point;
        }

        return total;
    }

    public Vector hadamard(Vector v) {
        Vector answer = new Vector(v.size());

        for (int i = 0; i < v.size(); i++) {
            answer.setX(i, points[i] * v.x(i));
        }

        return answer;
    }

    public void reset() {
        Arrays.fill(points, 0);
    }

    public void add(Vector v) {
        for (int i = 0; i < points.length; i++) {
            points[i] += v.x(i);
        }
    }

    public void randomise() {
        Random random = new Random();

        for (int i = 0; i < points.length; i++) {
            points[i] = random.nextDouble() + 0.0001f;
        }
    }

    public Vector divide(Vector v, double eps) throws Exception {
        Vector answer = new Vector(v.size());

        if (v.size() != points.length) {
            throw new Exception("Cam't divide");
        }

        for (int i = 0; i < answer.size(); i++) {
            answer.setX(i, points[i] / (v.x(i) + eps));
        }

        return answer;
    }

    public Vector square() {
        for (int i = 0; i < points.length; i++) {
            points[i] *= points[i];
        }

        return this;
    }

    public Vector sqrt() {
        Vector v = new Vector(points.length);

        for (int i = 0; i < v.size(); i++) {
            v.setX(i, Math.sqrt(points[i]));
        }

        return v;
    }

    public void concat(Vector v) {
        double[] newVec = new double[size() + v.size()];

        System.arraycopy(points, 0, newVec, 0, points.length);

        for (int i = points.length; i < newVec.length; i++) {
            newVec[i] = v.x(i - points.length);
        }

        points = newVec;
    }

    public void normalise() {
        float length = 0.0f;

        for (double point : points) {
            length += point * point;
        }

        length = (float) Math.sqrt(length);

        if (length == 0) {
            return;
        }

        for (int i = 0; i < points.length; i++) {
            points[i] /= length;
        }
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();

        for (double point : points) {
            builder.append(point).append("\t");
        }

        return builder.toString();
    }
}
