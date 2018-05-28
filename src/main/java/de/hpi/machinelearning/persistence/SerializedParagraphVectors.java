package de.hpi.machinelearning.persistence;


import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.springframework.data.annotation.Id;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;

@Getter
@JsonIgnoreProperties(ignoreUnknown = true)
public class SerializedParagraphVectors {

    @Id private final String networkType;
    private final byte[] serializedNeuralNetwork;

    public SerializedParagraphVectors(ParagraphVectors vectors, String type) throws IOException {
        OutputStream out = new ByteArrayOutputStream(100000000);
        WordVectorSerializer.writeParagraphVectors(vectors, out);

        this.serializedNeuralNetwork = ((ByteArrayOutputStream) out).toByteArray();
        this.networkType = type;
    }
}
