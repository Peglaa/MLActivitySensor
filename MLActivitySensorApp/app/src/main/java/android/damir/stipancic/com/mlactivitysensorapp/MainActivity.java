package android.damir.stipancic.com.mlactivitysensorapp;

import androidx.appcompat.app.AppCompatActivity;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import java.lang.annotation.Inherited;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity  implements SensorEventListener {

    private static final int TIME_STAMP = 100;

    private static List<Float> ax, ay, az;
    private static List<Float> gx, gy, gz;
    private static List<Float> lx, ly, lz;

    private SensorManager mSensorManager;
    private Sensor mAccelerometer, mGyroscope, mLinearAcceleration;

    private float[] results;
    private ActivityClassifier classifier;

    private TextView mBikingTxt, mDownstairsTxt, mJoggingTxt, mSittingTxt, mStandingTxt, mUpstairsTxt, mWalkingTxt;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        setupLayout();

        ax = new ArrayList<>(); ay = new ArrayList<>(); az = new ArrayList<>();
        gx = new ArrayList<>(); gy = new ArrayList<>(); gz = new ArrayList<>();
        lx = new ArrayList<>(); ly = new ArrayList<>(); lz = new ArrayList<>();

        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mLinearAcceleration = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);

        classifier = new ActivityClassifier(getApplicationContext());

        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(this, mLinearAcceleration, SensorManager.SENSOR_DELAY_FASTEST);

    }

    private void setupLayout() {

        mBikingTxt = findViewById(R.id.txtBiking);
        mJoggingTxt = findViewById(R.id.txtJogging);
        mDownstairsTxt = findViewById(R.id.txtDownstairs);
        mSittingTxt = findViewById(R.id.txtSitting);
        mStandingTxt = findViewById(R.id.txtStanding);
        mUpstairsTxt = findViewById(R.id.txtUpstairs);
        mWalkingTxt = findViewById(R.id.txtWalking);
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        Sensor sensor = sensorEvent.sensor;
        if(sensor.getType() == Sensor.TYPE_ACCELEROMETER){
            ax.add(sensorEvent.values[0]);
            ay.add(sensorEvent.values[1]);
            az.add(sensorEvent.values[2]);
        } else if(sensor.getType() == Sensor.TYPE_GYROSCOPE){
            gx.add(sensorEvent.values[0]);
            gy.add(sensorEvent.values[1]);
            gz.add(sensorEvent.values[2]);
        } else{
            lx.add(sensorEvent.values[0]);
            ly.add(sensorEvent.values[1]);
            lz.add(sensorEvent.values[2]);
        }

        predictActivity();
    }

    private void predictActivity() {

        List<Float> data = new ArrayList<>();
        if(ax.size() >= TIME_STAMP && ay.size() >= TIME_STAMP && az.size() >= TIME_STAMP
        && gx.size() >= TIME_STAMP && gy.size() >= TIME_STAMP && gz.size() >= TIME_STAMP
        && lx.size() >= TIME_STAMP && ly.size() >= TIME_STAMP && lz.size() >= TIME_STAMP){
            data.addAll(ax.subList(0, TIME_STAMP));
            data.addAll(ay.subList(0, TIME_STAMP));
            data.addAll(az.subList(0, TIME_STAMP));

            data.addAll(gx.subList(0, TIME_STAMP));
            data.addAll(gy.subList(0, TIME_STAMP));
            data.addAll(gz.subList(0, TIME_STAMP));

            data.addAll(lx.subList(0, TIME_STAMP));
            data.addAll(ly.subList(0, TIME_STAMP));
            data.addAll(lz.subList(0, TIME_STAMP));

            results = classifier.predictProbabilities(toFloatArray(data));
            Log.d("TAG", "predictActivity: " + Arrays.toString(results));

            mBikingTxt.setText("Biking: \t" + round(results[0]));
            mDownstairsTxt.setText("Downstairs: \t" + round(results[1]));
            mJoggingTxt.setText("Jogging: \t" + round(results[2]));
            mSittingTxt.setText("Sitting: \t" + round(results[3]));
            mStandingTxt.setText("Standing: \t" + round(results[4]));
            mUpstairsTxt.setText("Upstairs: \t" + round(results[5]));
            mWalkingTxt.setText("Walking: \t" + round(results[6]));

            ax.clear(); ay.clear(); az.clear();
            gx.clear(); gy.clear(); gz.clear();
            lx.clear(); ly.clear(); lz.clear();
        }
    }

    private float[] toFloatArray(List<Float> data){
        int i=0;
        float[] array = new float[data.size()];
        for(Float f : data)
            array[i++] = (f != null ? f: Float.NaN);

        return array;
    }

    private float round(float value){
        int decimal_places = 2;
        BigDecimal bigDecimal = new BigDecimal(Float.toString(value));
        bigDecimal = bigDecimal.setScale(decimal_places, BigDecimal.ROUND_HALF_UP);
        return bigDecimal.floatValue();
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    @Override
    protected void onResume() {
        super.onResume();

        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(this, mLinearAcceleration, SensorManager.SENSOR_DELAY_FASTEST);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        mSensorManager.unregisterListener(this);
    }
}