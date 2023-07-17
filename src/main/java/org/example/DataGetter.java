package org.example;

public interface DataGetter<X> {
    X at(int i);
    int size();
}