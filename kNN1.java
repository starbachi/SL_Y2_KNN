import java.util.ArrayList;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;



public class kNN1 {

    public static void main(String[] args) throws FileNotFoundException {
        readFile();
    }




    public static ArrayList<String> readFile() throws FileNotFoundException {

        ArrayList<String> dataArr = new ArrayList<String>();
        File file = new File("C://Users//erudo//OneDrive - University of Kent//Year 2//Autumn Term//CO5280 - Introduction to Artifical Intelligence//ai-a2//train_data.txt");
        Scanner fileReader = new Scanner(file);

        while (fileReader.hasNext()) {
            dataArr.add(fileReader.next());  
            System.out.println(fileReader.next());  
        }
        fileReader.close();
        
        return dataArr;
    }
}





    
