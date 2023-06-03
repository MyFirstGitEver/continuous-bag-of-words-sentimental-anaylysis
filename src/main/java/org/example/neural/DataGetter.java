package org.example.neural;

public interface DataGetter<X> {
    X at(int i);
    int size();
}
