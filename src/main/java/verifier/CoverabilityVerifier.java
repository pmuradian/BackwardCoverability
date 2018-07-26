package verifier;

//        You will have to implement the backward coverability algorithm for VAS (Vector
//        addition system). In the paper below, it is described for Petri Nets, which are
//        an equivalent formalism. http://www.lsv.fr/~haase/documents/bfhh.pdf
//        I want that sets of minimal elements describing upward-closed sets are stored
//        as RDDs. (RDD-look into Spark documentation)
//        The input file format looks as follows:
//        < vector1 >
//        < vector2 >
//        .
//        .
//        .
//        < vector i + 2 >
//        Each vector is a sequence of integer numbers separated with commas, there
//        is no comma after the last number.
//        First and second vectors are initial and final configurations, respectively.
//        (They should be positive in all coordinates)
//        Vectors from 3 to i+2 forms a VAS. You may assume that input data are
//        correct.
//        The program should print 1 if the finial configuration is coverable and 0 if
//        it is not coverable from the initial configuration.

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;

class VASVector implements Serializable {
    ArrayList<Integer> values;
    private int size = 0;

    VASVector(ArrayList<Integer> values) {
        this.values = values;
        size = this.values.size();
    }

    @Override
    public int hashCode() {
        Integer hash = 0;
        for (Integer v: values) {
            hash += v.hashCode();
        }
        return hash;
    }

    @Override
    public boolean equals(Object obj) {
        return this.hashCode() == obj.hashCode();
    }

    int getSize() {
        return size;
    }

    Integer valueAt(int index) {
        return values.get(index);
    }

    boolean isBiggerOrEqual(VASVector v) {
        for (int i = 0; i < v.getSize(); i++) {
            if (this.values.get(i) < v.valueAt(i)) {
                return false;
            }
        }
        return true;
    }

    boolean isBigger(VASVector v) {
        for (int i = 0; i < v.getSize(); i++) {
            if (values.get(i) <= v.valueAt(i)) {
                return false;
            }
        }
        return true;
    }

    VASVector subtract(VASVector transition) {
        for (int i = 0; i < values.size(); i++) {
            values.set(i, values.get(i) - transition.valueAt(i));
        }
        return this;
    }

    boolean isEqual(VASVector v) {
        for (int i = 0; i < values.size(); i++) {
            if (values.get(i) - v.valueAt(i) != 0) {
                return false;
            }
        }
        return true;
    }
}

public class CoverabilityVerifier {

    public static void main(String[] args) {

        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        SparkConf conf = new SparkConf().setMaster("local").setAppName("Word Count");

        // Create a Java version of the Spark Context
        JavaSparkContext context = new JavaSparkContext(conf);

        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example")
                .config("spark.some.config.option", "some-value")
                .getOrCreate();

        ArrayList<VASVector> inputVectors = new ArrayList<>();
        VASVector m0 = null;
        VASVector m = null;

        try (Stream<String> stream = Files.lines(Paths.get("input.txt"))) {
            int i = 0;
            Iterator<String> it = stream.iterator();
            while (it.hasNext()) {
                String line = it.next();
                String[] values = line.split(",");
                ArrayList<Integer> l = new ArrayList<>();
                for (int j = 0; j < values.length; j++) {
                    l.add(Integer.valueOf(values[j]));
                }
                if (i == 0) {
                    m0 = new VASVector(l);
                } else if (i == 1) {
                    m = new VASVector(l);
                } else {
                    inputVectors.add(new VASVector(l));
                }
                i++;
            }
        } catch (Exception e) {
            System.out.println("Error reading input file");
            return;
        }

        ArrayList<VASVector> l = new ArrayList<>();
        l.add(m);
        JavaRDD<VASVector> M = context.parallelize(l);
        l = new ArrayList<>();
        l.add(m0);
        JavaRDD<VASVector> m_0 = context.parallelize(l);
        JavaRDD<VASVector> configurations = m_0.union(M);

        final VASVector initial = m0;
        // while m0 is not contained in upwardClosure(M)
        while (upwardClosure(M, configurations).filter(v -> initial.isEqual(v)).isEmpty()) {
            JavaRDD<VASVector> B = pb(M, inputVectors).subtract(upwardClosure(M, configurations));
            if (B.isEmpty()) {
                System.out.println("0");
                return;
            }
            M = minbase(M.union(B));
            configurations = configurations.union(M);
        }
        System.out.println("1");
        context.stop();
    }

    private static JavaRDD<VASVector> upwardClosure(JavaRDD<VASVector> m, JavaRDD<VASVector> configurations) {
        JavaRDD<VASVector> ret = null;
        List<VASVector> vectors = m.collect();
        for (int i = 0; i < vectors.size(); i++) {
            int index = i;
            if (i == 0) {
                ret = configurations.filter(c -> c.isBiggerOrEqual(vectors.get(index)));
            } else {
                ret = ret.union(configurations.filter(c -> c.isBiggerOrEqual(vectors.get(index))));
            }
        }
        return ret.distinct();
    }

    private static JavaRDD<VASVector> minbase(JavaRDD<VASVector> vector) {
        Iterator<VASVector> it = vector.collect().iterator();
        JavaRDD<VASVector> ret = vector;
        while (it.hasNext()) {
            VASVector vec = it.next();
            JavaRDD<VASVector> notMinimal = ret.filter(v -> v.isBigger(vec));
            Iterator<VASVector> iterator = notMinimal.collect().iterator();
            ret = ret.filter(v -> { while(iterator.hasNext()) {
                if (iterator.next().isEqual(v)) {
                    return false;
                }
            }
            return true;
            });
        }
        return ret.distinct();
    }

    private static JavaRDD<VASVector> pb(JavaRDD<VASVector> vector, ArrayList<VASVector> transitions) {
        JavaRDD<VASVector> ret = null;
        for (int i = 0; i < transitions.size(); i++) {
            int index = i;
            if (i == 0) {
                ret = vector.map(v -> v.subtract(transitions.get(index)));
            } else {
                ret = ret.union(vector.map(v -> v.subtract(transitions.get(index))));
            }
        }
        return ret.distinct();
    }

    private static void printRDD(JavaRDD<VASVector> rdd, String message) {
        System.out.println(message + " {");
        rdd.collect().forEach(v -> System.out.println(v.values));
        System.out.println("}");
    }
}