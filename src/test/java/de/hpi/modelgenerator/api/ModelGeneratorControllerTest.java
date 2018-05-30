package de.hpi.modelgenerator.api;

import de.hpi.modelgenerator.persistence.ClassifierTrainingState;
import de.hpi.modelgenerator.services.ModelGeneratorService;
import lombok.AccessLevel;
import lombok.Getter;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.context.junit4.SpringRunner;
import org.springframework.test.web.servlet.MockMvc;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.verify;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@RunWith(SpringRunner.class)
@WebMvcTest(secure = false)
@Getter(AccessLevel.PRIVATE)
public class ModelGeneratorControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private ModelGeneratorService service;


    @Test
    public void generateCategoryClassifier() throws Exception {
        doNothing().when(getService()).generateCategoryClassifier(any(ClassifierTrainingState.class));

        getMockMvc()
                .perform(post("/generateCategoryClassifier"))
                .andExpect(status().isOk());

        verify(getService()).generateCategoryClassifier(any(ClassifierTrainingState.class));
    }

    @Test
    public void generateBrandClassifier() throws Exception {
        doNothing().when(getService()).generateBrandClassifier(any(ClassifierTrainingState.class));

        getMockMvc()
                .perform(post("/generateBrandClassifier"))
                .andExpect(status().isOk());

        verify(getService()).generateBrandClassifier(any(ClassifierTrainingState.class));
    }

    @Test
    public void generateModel() throws Exception {
        doNothing().when(getService()).generateModel(any(ClassifierTrainingState.class));

        getMockMvc()
                .perform(post("/generateModel"))
                .andExpect(status().isOk());

        verify(getService()).generateModel(any(ClassifierTrainingState.class));
    }

    @Test
    public void generateAllClassifiers() throws Exception {
        doNothing().when(getService()).generateCategoryClassifier(any(ClassifierTrainingState.class));
        doNothing().when(getService()).generateBrandClassifier(any(ClassifierTrainingState.class));
        doNothing().when(getService()).generateModel(any(ClassifierTrainingState.class));
        doNothing().when(getService()).freeTestingSet();

        getMockMvc()
                .perform(post("/generateAllClassifiers"))
                .andExpect(status().isOk());

        verify(getService()).generateCategoryClassifier(any(ClassifierTrainingState.class));
        verify(getService()).generateBrandClassifier(any(ClassifierTrainingState.class));
        verify(getService()).generateModel(any(ClassifierTrainingState.class));
    }
}