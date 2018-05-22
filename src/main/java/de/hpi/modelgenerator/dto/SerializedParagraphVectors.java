package de.hpi.modelgenerator.dto;


import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.AccessLevel;
import lombok.Setter;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;

@Setter(AccessLevel.PRIVATE)
@JsonIgnoreProperties(ignoreUnknown = true)
public class SerializedParagraphVectors {

    private byte[] serializedNeuralNetwork;

    public SerializedParagraphVectors(ParagraphVectors vectors) {
        OutputStream out = new ByteArrayOutputStream(40000000);
        try {
            WordVectorSerializer.writeParagraphVectors(vectors, out);
        } catch (IOException e) {
            e.printStackTrace();
        }

        setSerializedNeuralNetwork(((ByteArrayOutputStream) out).toByteArray());
    }
}
