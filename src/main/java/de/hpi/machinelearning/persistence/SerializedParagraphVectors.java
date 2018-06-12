package de.hpi.machinelearning.persistence;


import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.springframework.data.annotation.Id;

import java.io.*;

@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true)
@NoArgsConstructor
public class SerializedParagraphVectors {

    @Id private String networkType;
    private byte[] serializedNeuralNetwork;

    @JsonIgnore
    public SerializedParagraphVectors(ParagraphVectors vectors, String type) throws IOException {
        OutputStream out = new ByteArrayOutputStream(100000000);
        WordVectorSerializer.writeParagraphVectors(vectors, out);

        this.serializedNeuralNetwork = ((ByteArrayOutputStream) out).toByteArray();
        this.networkType = type;
    }

    @JsonIgnore
    public ParagraphVectors getNeuralNetwork() throws IOException {
        InputStream in = new ByteArrayInputStream(getSerializedNeuralNetwork());
        return VectorSerializer.readParagraphVectors(in);
    }
}
