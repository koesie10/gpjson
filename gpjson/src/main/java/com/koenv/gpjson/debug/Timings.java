package com.koenv.gpjson.debug;

import com.oracle.truffle.api.utilities.JSONHelper;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class Timings {
    public static final Timings TIMINGS = new Timings();

    private final Stack<Data> stack = new Stack<>();
    private final Map<String, Frame> frames = new HashMap<>();
    private final AtomicInteger frameId = new AtomicInteger(0);
    private final List<FrameEvent> events = new ArrayList<>();
    private final long startTime = System.nanoTime();

    public void start(String name) {
        start(name, null);
    }

    public void start(String name, String argument) {
        long time = System.nanoTime();
        Data data = new Data(name, argument);

        stack.push(data);

        Frame frame = frames.computeIfAbsent(data.getCombinedName(), frameName -> new Frame(frameId.getAndIncrement(), frameName));
        FrameEvent frameEvent = new FrameEvent("O", time - startTime, frame.id);
        events.add(frameEvent);
    }

    public void startNext(String name) {
        startNext(name, null);
    }

    public void startNext(String name, String argument) {
        end();
        start(name, argument);
    }

    public void end() {
        long time = System.nanoTime();
        Data data = stack.pop();

        Frame frame = frames.get(data.getCombinedName());
        FrameEvent frameEvent = new FrameEvent("C", time - startTime, frame.id);
        events.add(frameEvent);
    }

    public String export() {
        JSONHelper.JSONArrayBuilder framesBuilder = JSONHelper.array();
        frames.values().stream().sorted(Comparator.comparingInt(o -> o.id)).forEachOrdered(frame -> {
            JSONHelper.JSONObjectBuilder frameBuilder = JSONHelper.object();
            frameBuilder.add("name", frame.name);
            framesBuilder.add(frameBuilder);
        });

        JSONHelper.JSONObjectBuilder sharedBuilder = JSONHelper.object();
        sharedBuilder.add("frames", framesBuilder);

        JSONHelper.JSONArrayBuilder eventsBuilder = JSONHelper.array();
        events.forEach(event -> {
            JSONHelper.JSONObjectBuilder eventBuilder = JSONHelper.object();
            eventBuilder.add("type", event.type);
            eventBuilder.add("at", (int) (event.at / 1000));
            eventBuilder.add("frame", event.frame);
            eventsBuilder.add(eventBuilder);
        });

        JSONHelper.JSONObjectBuilder profileBuilder = JSONHelper.object();
        profileBuilder.add("type", "evented");
        profileBuilder.add("name", "profile");
        profileBuilder.add("unit", "microseconds");
        profileBuilder.add("startValue", 0);
        profileBuilder.add("endValue", (int) (events.get(events.size() - 1).at / 1000));
        profileBuilder.add("events", eventsBuilder);

        JSONHelper.JSONArrayBuilder profilesBuilder = JSONHelper.array();
        profilesBuilder.add(profileBuilder);

        JSONHelper.JSONObjectBuilder builder = JSONHelper.object();
        builder.add("version", "0.0.1");
        builder.add("$schema", "https://www.speedscope.app/file-format-schema.json");
        builder.add("shared", sharedBuilder);
        builder.add("profiles", profilesBuilder);

        return builder.toString();
    }

    private static class Data {
        final String name;
        final String argument;

        public Data(String name, String argument) {
            this.name = name;
            this.argument = argument;
        }

        public String getCombinedName() {
            if (argument == null) {
                return name;
            }

            return name + " (" + argument + ")";
        }
    }

    private static class Frame {
        private final int id;
        private final String name;

        public Frame(int id, String name) {
            this.id = id;
            this.name = name;
        }
    }

    private static class FrameEvent {
        private final String type;
        private final long at;
        private final int frame;

        public FrameEvent(String type, long at, int frame) {
            this.type = type;
            this.at = at;
            this.frame = frame;
        }
    }
}
