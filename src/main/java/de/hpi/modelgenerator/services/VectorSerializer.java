package de.hpi.modelgenerator.services;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.output.CloseShieldOutputStream;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import static org.deeplearning4j.models.embeddings.loader.WordVectorSerializer.encodeB64;
import static org.deeplearning4j.models.embeddings.loader.WordVectorSerializer.writeWordVectors;

public class VectorSerializer {

    public static void writeParagraphVectors(ParagraphVectors vectors, OutputStream stream) throws IOException {
        ZipOutputStream zipfile = new ZipOutputStream(new BufferedOutputStream(new CloseShieldOutputStream(stream)));

        ZipEntry syn0 = new ZipEntry("syn0.txt");
        zipfile.putNextEntry(syn0);

        // writing out syn0
        File tempFileSyn0 = File.createTempFile("paravec", "0");
        File tempFileSyn1 = File.createTempFile("paravec", "1");
        File tempFileCodes = File.createTempFile("paravec", "c");
        File tempFileHuffman = File.createTempFile("paravec", "h");
        File tempFileFreqs = File.createTempFile("paravec", "f");
        tempFileSyn0.deleteOnExit();
        tempFileSyn1.deleteOnExit();
        tempFileCodes.deleteOnExit();
        tempFileHuffman.deleteOnExit();
        tempFileFreqs.deleteOnExit();

        try {

            writeWordVectors(vectors.lookupTable(), tempFileSyn0);

            FileUtils.copyFile(tempFileSyn0, zipfile);

            // writing out syn1
            INDArray syn1 = ((InMemoryLookupTable<VocabWord>) vectors.getLookupTable()).getSyn1();

            if (syn1 != null)
                try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileSyn1))) {
                    for (int x = 0; x < syn1.rows(); x++) {
                        INDArray row = syn1.getRow(x);
                        StringBuilder builder = new StringBuilder();
                        for (int i = 0; i < row.length(); i++) {
                            builder.append(row.getDouble(i)).append(" ");
                        }
                        writer.println(builder.toString().trim());
                    }
                }

            ZipEntry zSyn1 = new ZipEntry("syn1.txt");
            zipfile.putNextEntry(zSyn1);

            FileUtils.copyFile(tempFileSyn1, zipfile);

            ZipEntry hC = new ZipEntry("codes.txt");
            zipfile.putNextEntry(hC);

            // writing out huffman tree
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileCodes))) {
                for (int i = 0; i < vectors.getVocab().numWords(); i++) {
                    VocabWord word = vectors.getVocab().elementAtIndex(i);
                    StringBuilder builder = new StringBuilder(encodeB64(word.getLabel())).append(" ");
                    for (int code : word.getCodes()) {
                        builder.append(code).append(" ");
                    }

                    writer.println(builder.toString().trim());
                }
            }

            FileUtils.copyFile(tempFileCodes, zipfile);

            ZipEntry hP = new ZipEntry("huffman.txt");
            zipfile.putNextEntry(hP);

            // writing out huffman tree
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileHuffman))) {
                for (int i = 0; i < vectors.getVocab().numWords(); i++) {
                    VocabWord word = vectors.getVocab().elementAtIndex(i);
                    StringBuilder builder = new StringBuilder(encodeB64(word.getLabel())).append(" ");
                    for (int point : word.getPoints()) {
                        builder.append(point).append(" ");
                    }

                    writer.println(builder.toString().trim());
                }
            }

            FileUtils.copyFile(tempFileHuffman, zipfile);

            ZipEntry config = new ZipEntry("config.json");
            zipfile.putNextEntry(config);
            IOUtils.write(vectors.getConfiguration().toJson(), zipfile, StandardCharsets.UTF_8);


            ZipEntry labels = new ZipEntry("labels.txt");
            zipfile.putNextEntry(labels);
            StringBuilder builder = new StringBuilder();
            for (VocabWord word : vectors.getVocab().tokens()) {
                if (word.isLabel())
                    builder.append(encodeB64(word.getLabel())).append("\n");
            }
            IOUtils.write(builder.toString().trim(), zipfile, StandardCharsets.UTF_8);

            ZipEntry hF = new ZipEntry("frequencies.txt");
            zipfile.putNextEntry(hF);

            // writing out word frequencies
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileFreqs))) {
                for (int i = 0; i < vectors.getVocab().numWords(); i++) {
                    VocabWord word = vectors.getVocab().elementAtIndex(i);
                    builder = new StringBuilder(encodeB64(word.getLabel())).append(" ").append(word.getElementFrequency())
                            .append(" ").append(vectors.getVocab().docAppearedIn(word.getLabel()));

                    writer.println(builder.toString().trim());
                }
            }

            FileUtils.copyFile(tempFileFreqs, zipfile);

            zipfile.flush();
            zipfile.close();
        } finally {
            for(File f : new File[]{tempFileSyn0, tempFileSyn1, tempFileCodes, tempFileHuffman, tempFileFreqs}){
                try{
                    f.delete();
                } catch (Exception e){
                    //Ignore, is temp file
                }
            }
        }
    }

}
