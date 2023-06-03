package org.example;

import org.example.neural.*;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Main {
    static String prefix = "D:\\Source code\\Outer data\\BOW\\word2vec-nlp-tutorial\\labeledTrainData\\";

    public static void main(String[] args) throws Exception {
        //train(600);
        //secondLayer(600);
        test(350);
    }

    private static void test(int featureNum) throws IOException {
        HashMap<String, Integer> vocab = loadVocab();

        SimpleNeuralNetwork model = new SimpleNeuralNetwork(new DenseLayer[]{
                new DenseLayer(new ReluActivation(vocab.size(), featureNum)),
                new DenseLayer(new SoftMaxActivation(featureNum, vocab.size()))
        }, "CBOW on 50 reviews");
        Vector[] wordEmbeddings = model.w(1);

        SimpleNeuralNetwork trainer = new SimpleNeuralNetwork(new DenseLayer[] {
                new DenseLayer(new ReluActivation(featureNum, 20)),
                new DenseLayer(new SoftMaxActivation(20, 2))
        }, new CrossEntropy(), null, null);

        trainer.loadParams("sentiment"); // load only
        Pair<Vector, Integer>[] frames = loadReviews(vocab, wordEmbeddings, "5_000 tests.txt", featureNum);

        int hit = 0, timer = 0;
        for(Pair<Vector, Integer> frame :frames) {
            Vector pred = trainer.predict(frame.first);

            if(pred.x(0) > pred.x(1) && frame.second == 0) {
                hit++;
            }
            else if(pred.x(0) < pred.x(1) && frame.second == 1) {
                hit++;
            }

            timer++;

            if(timer % 100 == 0) {
                System.out.println(timer + " tests done. Hit: " + hit);
            }
        }

        System.out.println((hit / (float)frames.length * 100) + " %");
    }

    private static void secondLayer(int featureNum) throws Exception {
        HashMap<String, Integer> vocab = loadVocab();

        SimpleNeuralNetwork model = new SimpleNeuralNetwork(new DenseLayer[]{
                new DenseLayer(new ReluActivation(vocab.size(), featureNum)),
                new DenseLayer(new SoftMaxActivation(featureNum, vocab.size()))
        }, "CBOW on 50 reviews");
        Vector[] wordEmbeddings = model.w(1);

        Pair<Vector, Integer>[] frames = loadReviews(vocab, wordEmbeddings, " 10_000 reviews.txt", featureNum);

        DataGetter<Vector> xGetter = new DataGetter<>() {
            @Override
            public Vector at(int i) {
                return frames[i].first;
            }

            @Override
            public int size() {
                return frames.length;
            }
        };

        DataGetter<Vector> yGetter = new DataGetter<>() {
            @Override
            public Vector at(int i) {
                Vector v = new Vector(2);
                v.setX(frames[i].second, 1);

                return v;
            }

            @Override
            public int size() {
                return frames.length;
            }
        };

        SimpleNeuralNetwork trainer = new SimpleNeuralNetwork(new DenseLayer[] {
                new DenseLayer(new ReluActivation(featureNum, 20)),
                new DenseLayer(new SoftMaxActivation(20, 2))
        }, new CrossEntropy(), xGetter, yGetter);


        //trainer.loadParams("sentiment");
        trainer.train(0.001, 1_000, 100, 10, "sentiment", true);
    }

    private static Pair<Vector, Integer>[] loadReviews(
            HashMap<String, Integer> vocab,
            Vector[] embeddings, String path, int featureNum) throws IOException {
        FileInputStream fIn = new FileInputStream(prefix + path);

        String data = new String(fIn.readAllBytes());

        String[] reviews = data.split("\3");

        Pair<Vector, Integer>[] frames = new Pair[reviews.length];

        for(int i=0;i<reviews.length;i++) {
            String[] info = reviews[i].split("\2");
            frames[i] = new Pair<>(reviewToVec(info[0], vocab, embeddings, featureNum), Integer.parseInt(info[1]));

            if(i % 100 == 0) {
                System.out.println(i + " / " + reviews.length + " scanned");
            }
        }

        fIn.close();

        return frames;
    }

    static Vector reviewToVec(
            String review,
            HashMap<String, Integer> vocab,
            Vector[] embeddings, int featureNum) throws IOException {
        List<String> tokens = TextProcessing.lemmas(review);
        TextProcessing.removeStopWordsAndWeirdStrings(tokens, false, true, false);

        Vector v = new Vector(featureNum);
        for(String token : tokens) {
            if(vocab.containsKey(token)) {
                v.add(embeddings[vocab.get(token)]);
            }
        }

        return v.scaleBy(1.0 / tokens.size());
    }

    static void train(int featureNum) throws Exception {
        List<Vector> contextVecs = loadVecs("vecs-50.txt");
        List<String> centerWords = loadCenterWords();
        HashMap<String, Integer> vocab = loadVocab();

        DataGetter<Vector> xGetter = new DataGetter<Vector>() {
            @Override
            public Vector at(int i) {
                return contextVecs.get(i);
            }

            @Override
            public int size() {
                return contextVecs.size();
            }
        };

        DataGetter<Vector> yGetter = new DataGetter<Vector>() {
            @Override
            public Vector at(int i) {
                Vector v = new Vector(vocab.size());

                String word = centerWords.get(i);
                v.setX(vocab.get(word), 1);

                return v; // one-hot vector
            }

            @Override
            public int size() {
                return vocab.size();
            }
        };


        SimpleNeuralNetwork network = new SimpleNeuralNetwork(new DenseLayer[]{
                new DenseLayer(new ReluActivation(vocab.size(), featureNum)),
                new DenseLayer(new SoftMaxActivation(featureNum, vocab.size()))
        }, new CrossEntropy(),  xGetter, yGetter);

        //network.loadParams("CBOW on 50 reviews");
        network.train(0.01f,   5, 400, 1,"CBOW on 50 reviews", false);
    }

    static HashMap<String, Integer> loadVocab() throws IOException {
        BufferedReader reader = Files.newBufferedReader(Paths.get(prefix + "vocab-50.txt"));

        String line;
        HashMap<String, Integer> hm = new HashMap<>();

        while((line = reader.readLine()) != null) {
            String[] data = line.split("\t");

            hm.put(data[0], Integer.parseInt(data[1]));
        }

        reader.close();

        return hm;
    }

    static List<String> loadCenterWords() throws IOException {
        BufferedReader reader = Files.newBufferedReader(Paths.get(prefix + "centers-50.txt"), Charset.forName("Cp1252"));

        String line;
        List<String> centerWords = new ArrayList<>();

        while((line = reader.readLine()) != null) {
            centerWords.add(line);
        }

        reader.close();

        return centerWords;
    }

    static List<Vector> loadVecs(String path) throws IOException {
        BufferedReader reader = Files.newBufferedReader(Paths.get(prefix + path));

        String line;
        List<Vector> contextVecs = new ArrayList<>();

        while((line = reader.readLine()) != null) {
            contextVecs.add(new Vector(line, ", "));
        }

        reader.close();

        return contextVecs;
    }
}