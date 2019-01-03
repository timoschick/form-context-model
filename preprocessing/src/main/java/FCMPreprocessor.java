import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class FCMPreprocessor {

    private static Pattern sentencePattern = Pattern.compile(" \\. ?");
    private static Pattern wordPattern = Pattern.compile("\\s+");

    private static Set<String> wordsToIgnore = new HashSet<>(Arrays.asList("-rrb-", "-lrb-", "-rsb-", "-lsb-"));

    private static Pattern wordsToIgnorePattern = Pattern.compile("[^A-Za-z0-9%§$€]+");
    private static Matcher wordsToIgnoreMatcher = wordsToIgnorePattern.matcher("");

    private Map<String, Integer> word2Id;
    private Map<Integer, String> id2Word;

    private Set<String> wordVocab;
    private Map<Integer, Entry> dict;

    private Properties properties;

    private FCMPreprocessor() {
        dict = new HashMap<>();
        wordVocab = new HashSet<>();
        word2Id = new HashMap<>();
        id2Word = new HashMap<>();
    }

    public static void main(String[] args) throws IOException {

        ArgumentParser parser = ArgumentParsers.newFor(FCMPreprocessor.class.getSimpleName()).build()
                .defaultHelp(true)
                .description("Preprocess a corpus of sentences for the FCM to be trained on.");

        parser.addArgument("--input", "-i").required(true).help("the input corpus");
        parser.addArgument("--output", "-o").required(true).help("the output directory");
        parser.addArgument("--vocab", "-v").required(true).help("The vocab file to be used. Each line must contain exactly one word. For each context, every word that is not in this file is replaced by an UNK token");
        parser.addArgument("--word_vocab", "-wv").help("The words for which contexts are to be created, in the same format as the vocab file. If not specified, this file is assumed to be the same as the regular vocab file");

        parser.addArgument("--buckets", "-b").setDefault(25).help("The number of context buckets to create. Lower values take less time but more RAM");
        parser.addArgument("--context", "-c").setDefault(25).help("The maximum number of words to the left and right of each word to consider for each context");

        Namespace ns = null;
        try {
            ns = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }

        String trainFile = ns.getString("input");
        String contextVocabFile = ns.getString("vocab");

        String wordVocabFile = ns.getString("word_vocab");
        if(wordVocabFile == null || wordVocabFile.isEmpty()) {
            wordVocabFile = contextVocabFile;
        }

        String outputDir = ns.getString("output") + File.separator;
        Random rand = new Random(42);

        int numBuckets = ns.getInt("buckets");

        Properties properties = new Properties();
        properties.maxContextSize = ns.getInt("context");
        properties.maxWordCount = 1000000;
        properties.maxSentencesPerWord = 1000;
        properties.keepWordsWithoutContexts = true;

        bucketizeVocab(wordVocabFile, numBuckets, rand);

        for (int i = 0; i < numBuckets; i++) {
            System.out.println("Building Context Dictionary for bucket " + i);
            String bucketFile = wordVocabFile + ".bucket" + i;
            FCMPreprocessor cd = FCMPreprocessor.fromFile(trainFile, bucketFile, contextVocabFile, properties);
            cd.toFile(outputDir + "train.bucket" + i + ".txt");
            cd = null;
            System.gc();
        }
    }

    private static void bucketizeVocab(String vocabFile, int nrOfBuckets, Random rand) throws IOException {

        List<String> lines = Files.readAllLines(Paths.get(vocabFile));
        // remove duplicates
        lines = new ArrayList<>(new HashSet<>(lines));
        Collections.shuffle(lines, rand);

        int linesPerBucket = (lines.size() / nrOfBuckets) + 1;

        for (int i = 0; i < nrOfBuckets; i++) {

            int from = Math.min(lines.size(), i * linesPerBucket);
            int to = Math.min(lines.size(), (i + 1) * linesPerBucket);
            List<String> bucketContent = lines.subList(from, to);
            Files.write(Paths.get(vocabFile + ".bucket" + i), bucketContent);
        }
    }

    private void toFile(String file) throws IOException {

        try (FileWriter writer = new FileWriter(new File(file))) {

            for (int wordId : dict.keySet()) {

                String word = id2Word.get(wordId);
                Entry dictEntry = dict.get(wordId);

                StringJoiner contextJoiner = new StringJoiner("\t");

                for (List<Integer> sentIds : dictEntry.sentences) {

                    String sent = String.join(" ", sentIds.stream().map(id -> id2Word.getOrDefault(id, properties.unkString)).collect(Collectors.toList()));
                    contextJoiner.add(sent);
                }

                String line = word + "\t" + contextJoiner.toString() + "\n";
                writer.write(line);
            }
            writer.flush();
        }

    }

    private static FCMPreprocessor fromFile(String trainFile, String wordVocabFile, String contextVocabFile, Properties properties) throws IOException {

        FCMPreprocessor ret = new FCMPreprocessor();
        ret.properties = properties;

        int id = 0;

        try (BufferedReader br = new BufferedReader(new FileReader(contextVocabFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                String word = line.split(" ", 2)[0];

                if (!isValid(word)) continue;
                ret.word2Id.put(word, id);
                ret.id2Word.put(id, word);
                id++;
            }
        }


        try (BufferedReader br = new BufferedReader(new FileReader(wordVocabFile))) {
            String line;
            while ((line = br.readLine()) != null) {

                String[] comps = line.split(" ", 2);

                String word;
                int count;

                if (comps.length == 2) {
                    word = comps[0];
                    try {
                        count = Integer.valueOf(comps[1]);
                    } catch (NumberFormatException e) {
                        count = 1;
                        System.out.println("Could not parse count " + comps[1]);
                    }
                } else {
                    word = comps[0];
                    count = 1;
                }

                if (!isValid(word) || count > properties.maxWordCount) {
                    System.out.println("Skipping invalid word " + word);
                    continue;
                }

                if (!ret.word2Id.containsKey(word)) {
                    ret.word2Id.put(word, id);
                    ret.id2Word.put(id, word);
                    id++;
                }

                ret.wordVocab.add(word);
                ret.dict.put(ret.word2Id.get(word), new Entry());
            }
        }

        long startTime = System.nanoTime();

        try (BufferedReader br = new BufferedReader(new FileReader(trainFile))) {
            int count = 0;
            String line;
            while ((line = br.readLine()) != null) {
                ret.addLine(line);
                count++;

                if (count % 100000 == 0) {
                    int secs = (int) ((System.nanoTime() - startTime) / 1e9);
                    System.out.println("done adding " + count + " lines, took " + secs + "s");
                }

                if (count == properties.maxLines) {
                    break;
                }
            }
        }

        return ret;
    }

    private static boolean isValid(String word) {
        if (wordsToIgnore.contains(word)) return false;

        wordsToIgnoreMatcher.reset(word);
        return !wordsToIgnoreMatcher.matches();
    }

    private void addLine(String line) {

        String[] sentences = sentencePattern.split(line);
        for (String sentence : sentences) {
            addSentence(sentence);
        }

    }

    private void addSentence(String sentence) {

        String[] words = wordPattern.split(sentence);

        List<Integer> knownWordIndices = new ArrayList<>();
        for (int i = 0; i < words.length; i++) {
            if (wordVocab.contains(words[i])) {
                knownWordIndices.add(i);
            }
        }

        List<Integer> wordIndices = new ArrayList<>();
        for (int i = 0; i < words.length; i++) {
            if (word2Id.containsKey(words[i])) {
                wordIndices.add(word2Id.get(words[i]));
            } else {
                wordIndices.add(properties.unkId);
            }
        }

        for (int i : knownWordIndices) {

            String word = words[i];
            Entry entry = dict.get(word2Id.get(word));

            int startIndex = properties.maxContextSize > 0 ? Math.max(0, i - properties.maxContextSize) : 0;
            int endIndex = properties.maxContextSize > 0 ? Math.min(words.length, i + properties.maxContextSize + 1) : words.length;

            if (entry.updateCount < properties.maxSentencesPerWord) {
                List<Integer> subsentence = wordIndices.subList(startIndex, endIndex);
                entry.add(subsentence);
            }
        }

    }

    static class Entry {

        Set<List<Integer>> sentences;
        int updateCount = 0;

        Entry() {
            sentences = new HashSet<>();
        }

        @Override
        public boolean equals(Object o) {
            if (o == null) return false;
            if (!(o instanceof Entry)) return false;
            Entry e = (Entry) o;
            return updateCount == e.updateCount && sentences.equals(e.sentences);
        }

        @Override
        public int hashCode() {
            return Integer.hashCode(updateCount) + 13 * sentences.hashCode();
        }

        void add(List<Integer> sentence) {
            sentences.add(sentence);
            updateCount++;
        }

        public String toString() {
            return sentences.size() + " Entries: " + sentences.toString();
        }

    }

    static class Properties {
        int maxContextSize = -1;
        int maxLines = -1;
        int maxSentencesPerWord = -1;
        int unkId = -1;
        int maxWordCount = Integer.MAX_VALUE;
        boolean keepWordsWithoutContexts = false;
        String unkString = "UNK";
    }
}
