package org.example.neural;

public class Matrix {
    private final double[][] entries;

    Matrix(Vector v, boolean columnVector) {
        if(columnVector) {
            entries = new double[v.size()][1];

            for(int i=0;i<v.size();i++) {
                entries[i][0] = v.x(i);
            }
        }
        else {
            entries = new double[1][v.size()];

            for(int i=0;i<v.size();i++) {
                entries[0][i] = v.x(i);
            }
        }
    }

    Matrix(double[]... data) {
        entries = data;
    }

    Matrix(int width, int height) {
        entries = new double[width][height];
    }

    public Vector[] mul(Matrix mat) throws Exception {
        Pair<Integer, Integer> shape = mat.shape();

        if(entries[0].length != shape.first) {
            throw new Exception("Can't multiply these two matrices");
        }

        int common = shape.first;
        Vector[] newMat = new Vector[entries.length];

        for(int i=0;i<entries.length;i++) {
            newMat[i] = new Vector(shape.second);

            for(int j=0;j<shape.second;j++) {
                double total = 0;

                for(int k=0;k<common;k++) {
                    total += entries[i][k] * mat.entries[k][j];
                }

                newMat[i].setX(j, total);
            }
        }

        return newMat;
    }

    public Pair<Integer, Integer> shape() {
        return new Pair<>(entries.length, entries[0].length);
    }

    public boolean identical(Vector[] mat) {
        if(entries.length != mat.length || entries[0].length != mat[0].size()) {
            return false;
        }

        for(int i=0;i<mat.length;i++) {
            for(int j=0;j<mat[0].size();j++) {
                if(mat[i].x(j) != entries[i][j]) {
                    return false;
                }
            }
        }

        return true;
    }

    public void reset() {
        for(int i=0;i<entries.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                entries[i][j] = 0;
            }
        }
    }

    public void add(Vector[] mat) throws Exception {
        if(entries.length != mat.length || entries[0].length != mat[0].size()) {
            throw new Exception("Can't add these two matrices");
        }
        for(int i=0;i<entries.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                entries[i][j] += mat[i].x(j);
            }
        }
    }

    public void add(Matrix matrix) throws Exception {
        Pair<Integer, Integer> shape = matrix.shape();

        if(entries.length != shape.first || entries[0].length != shape.second) {
            throw new Exception("Can't add these two matrices");
        }
        for(int i=0;i<entries.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                entries[i][j] += matrix.entries[i][j];
            }
        }
    }

    public Matrix square() {
        for(int i=0;i<entries.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                entries[i][j] *= entries[i][j];
            }
        }

        return this;
    }

    public Matrix copy() {
        double[][] newMat = new double[entries.length][entries[0].length];

        for(int i=0;i<newMat.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                newMat[i][j] = entries[i][j];
            }
        }

        return new Matrix(newMat);
    }

    public Vector[] vectorize(boolean byRow) {
        if(byRow) {
            Vector[] vectors = new Vector[entries.length];

            for(int i=0;i<entries.length;i++) {
                vectors[i] = new Vector(entries[i]);
            }

            return vectors;
        }

        Vector[] vectors = new Vector[entries[0].length];

        for(int i=0;i<entries[0].length;i++) {
            vectors[i] = new Vector(entries.length);

            for(int j=0;j<entries.length;j++) {
                vectors[i].setX(j, entries[j][i]);
            }
        }

        return vectors;
    }

    public Matrix scale(double scale) {
        for(int i=0;i<entries.length;i++) {
            for(int j=0;j<entries[0].length;j++) {
                entries[i][j]  *= scale;
            }
        }

        return this;
    }

    public static Matrix transpose(Vector[] mat) {
        double[][] answer = new double[mat[0].size()][mat.length];

        for(int i=0;i<mat.length;i++){
            for(int j=0;j<mat[0].size();j++) {
                answer[j][i] = mat[i].x(j);
            }
        }

        return new Matrix(answer);
    }
}