/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diuf.diva.dia.ms.util;

/**
 *
 * @author Mathias Seuret
 */
public class Random {
    private static java.util.Random rand;
    private Random() {}
    
    // Synchronized not needed in this context - if several threads
    // are running, it's likely the random numbers will anyway be
    // dispatched randomly to the threads
    private static java.util.Random instance() {
        if (rand==null) {
            rand = new java.util.Random(10191);
        }
        return rand;
    }
    
    public static void setSeed(long seed) {
        instance().setSeed(seed);
    }
    
    public static float nextFloat() {
        return instance().nextFloat();
    }
    
    public static double nextDouble() {
        return instance().nextDouble();
    }
    
    public static int nextInt() {
        return instance().nextInt();
    }
    
    public static int nextInt(int upperBound) {
        return instance().nextInt(upperBound);
    }
    
    public static boolean nextBoolean() {
        return instance().nextBoolean();
    }
}
