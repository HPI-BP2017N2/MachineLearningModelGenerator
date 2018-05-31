package de.hpi.modelgenerator.api;

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
        doNothing().when(getService()).generateCategoryClassifier();

        getMockMvc()
                .perform(post("/generateCategoryClassifier"))
                .andExpect(status().isOk());

        verify(getService()).generateCategoryClassifier();
    }

    @Test
    public void generateBrandClassifier() throws Exception {
        doNothing().when(getService()).generateBrandClassifier();

        getMockMvc()
                .perform(post("/generateBrandClassifier"))
                .andExpect(status().isOk());

        verify(getService()).generateBrandClassifier();
    }

    @Test
    public void generateModel() throws Exception {
        doNothing().when(getService()).generateModel();

        getMockMvc()
                .perform(post("/generateModel"))
                .andExpect(status().isOk());

        verify(getService()).generateModel();
    }

    @Test
    public void generateAllClassifiers() throws Exception {
        doNothing().when(getService()).generateCategoryClassifier();
        doNothing().when(getService()).generateBrandClassifier();
        doNothing().when(getService()).generateModel();
        doNothing().when(getService()).freeTestingSet();

        getMockMvc()
                .perform(post("/generateAllClassifiers"))
                .andExpect(status().isOk());

        verify(getService()).generateCategoryClassifier();
        verify(getService()).generateBrandClassifier();
        verify(getService()).generateModel();
    }
}