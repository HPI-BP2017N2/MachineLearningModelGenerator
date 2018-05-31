package de.hpi.machinelearning.persistence;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.documentiterator.LabelsSource;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.LinkedList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import static org.deeplearning4j.models.embeddings.loader.WordVectorSerializer.decodeB64;

class VectorSerializer {

    static ParagraphVectors readParagraphVectors(InputStream stream) throws IOException {
        File tmpFile = File.createTempFile("restore", "paravec");
        try {
            FileUtils.copyInputStreamToFile(stream, tmpFile);
            return readParagraphVectors(tmpFile);
        } finally {
            tmpFile.delete();
        }
    }

    private static ParagraphVectors readParagraphVectors(File file) throws IOException {
        Word2Vec w2v = WordVectorSerializer.readWord2Vec(file);

        List<String> labelsList = new LinkedList<>();
        try (ZipFile zipFile = new ZipFile(file)) {
            ZipEntry labels = zipFile.getEntry("labels.txt");

            if (labels != null) {
                InputStream stream = zipFile.getInputStream(labels);
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        VocabWord word = w2v.getVocab().tokenFor(decodeB64(line.trim()));
                        if (word != null) {
                            word.markAsLabel(true);
                            labelsList.add(decodeB64(line.trim()));
                        }
                    }
                }
            }
        }

        ParagraphVectors vectors = new ParagraphVectors.Builder(w2v.getConfiguration()).vocabCache(w2v.getVocab())
                .lookupTable(w2v.getLookupTable()).resetModel(false).labelsSource(new LabelsSource(labelsList)).build();
        vectors.extractLabels();

        return vectors;
    }

}