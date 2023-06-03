package org.example;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;

public class TextProcessing {
    static List<String> lemmas(String doc){
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma");

        StanfordCoreNLP pipeline;
        pipeline = new StanfordCoreNLP(props, false);
        Annotation document = pipeline.process(doc);

        List<String> lemmas = new ArrayList<>();
        for(CoreMap sentence: document.get(CoreAnnotations.SentencesAnnotation.class))
        {
            for(CoreLabel token: sentence.get(CoreAnnotations.TokensAnnotation.class))
            {
                String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);

                lemmas.add(lemma.toLowerCase());
            }
        }

        return lemmas;
    }

    static void removeStopWordsAndWeirdStrings(Collection<String> words,
                                               boolean removeStopwords,
                                               boolean removeWeird, boolean removeHashTag) throws IOException {
        Set<String> removed = new TreeSet<>();

        if(removeWeird) {
            for(String word : words) {
                if(word.length() > 1 && word.charAt(0) == '#') {
                    removed.add(word);

                    if(!removeHashTag) {
                        removed.add(word.substring(1));
                    }
                }

                if(!Pattern.compile("^[a-zA-Z]+").matcher(word).find()){
                    removed.add(word);
                }
            }
        }

        if(removeStopwords) {
            BufferedReader reader = new BufferedReader(new FileReader("english"));
            String line;

            while((line = reader.readLine()) != null) {
                if(words.contains(line)){
                    removed.add(line);
                }
            }
        }

        for(String word : removed) {
            words.removeIf(word::equals);
        }
    }
}
